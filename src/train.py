"""Pętla treningowa, walidacja krzyżowa i trening modelu końcowego.

Przykłady:
  python -m src.train cv --model efficientnet_b0
  python -m src.train final --model efficientnet_b0 --params out/metrics/optuna_best.json
"""
import argparse
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import (CLASSES, METRICS_DIR, MODELS_DIR, SEED, MRIDataset, load_images,
                      make_folds, training_table)
from src.metrics import summary
from src.models import build_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Domyślne hiperparametry przed strojeniem. CNN od zera potrzebuje więcej epok
# i większego LR, sieci wstępnie wytrenowane - mniej epok i mniejszego LR.
DEFAULTS = {
    'cnn': {'lr': 1e-3, 'weight_decay': 1e-4, 'dropout': 0.3, 'batch_size': 32, 'epochs': 30,
            'head_epochs': 0, 'n_unfreeze': 0},
    'efficientnet_b0': {'lr': 1e-3, 'weight_decay': 1e-4, 'dropout': 0.3, 'batch_size': 32, 'epochs': 15,
                        'head_epochs': 2, 'n_unfreeze': 4},
    'resnet50': {'lr': 1e-3, 'weight_decay': 1e-4, 'dropout': 0.3, 'batch_size': 32, 'epochs': 15,
                 'head_epochs': 2, 'n_unfreeze': 2},
}
# szkielet uczony z LR 10 razy mniejszym niż głowica, żeby nie zniszczyć wag z ImageNet
BACKBONE_LR_FACTOR = 0.1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model, name, p, stage):
    """stage 'head' - uczona tylko głowica; stage 'finetune' - głowica + ostatnie bloki szkieletu."""
    if name == 'cnn':
        return torch.optim.AdamW(model.parameters(), lr=p['lr'], weight_decay=p['weight_decay'])
    model.set_trainable(0 if stage == 'head' else p['n_unfreeze'])
    head = model.net.classifier if name == 'efficientnet_b0' else model.net.fc
    head_ids = {id(q) for q in head.parameters()}
    backbone = [q for q in model.net.parameters() if q.requires_grad and id(q) not in head_ids]
    groups = [{'params': list(head.parameters()), 'lr': p['lr']}]
    if backbone:
        groups.append({'params': backbone, 'lr': p['lr'] * BACKBONE_LR_FACTOR})
    return torch.optim.AdamW(groups, weight_decay=p['weight_decay'])


@torch.no_grad()
def predict(model, dataset, batch_size=64):
    """Prawdopodobieństwa klas (softmax) dla całego zbioru."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    out = []
    for x, _ in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
            logits = model(x.to(DEVICE))
        out.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(out).numpy()


def train_model(name, p, train_ds, val_ds=None, seed=SEED, log=print, on_epoch=None):
    """Trenuje model przez p['epochs'] epok.

    Jeśli podano val_ds, po każdej epoce liczymy macro-F1 na walidacji i zapamiętujemy
    wagi najlepszej epoki (zamiast ostatniej). Zwraca model, historię i najlepszą epokę."""
    set_seed(seed)
    model = build_model(name, p['dropout']).to(DEVICE)
    loader = DataLoader(train_ds, batch_size=p['batch_size'], shuffle=True, num_workers=6,
                        drop_last=True, persistent_workers=True)
    criterion = nn.CrossEntropyLoss()
    stage = 'head' if p['head_epochs'] > 0 else 'finetune'
    optimizer = make_optimizer(model, name, p, stage)
    # LR maleje kosinusowo od startu fine-tuningu do końca treningu
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p['epochs'] - p['head_epochs'])

    history, best = [], {'macro_f1': -1, 'epoch': 0, 'state': None}
    for epoch in range(1, p['epochs'] + 1):
        if stage == 'head' and epoch > p['head_epochs']:
            stage = 'finetune'
            optimizer = make_optimizer(model, name, p, stage)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p['epochs'] - p['head_epochs'])
        t0 = time.time()
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=DEVICE.type == 'cuda'):
                logits = model(x)
                loss = criterion(logits.float(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        if stage == 'finetune':
            scheduler.step()

        row = {'epoch': epoch, 'train_loss': total_loss / n, 'train_acc': correct / n}
        if val_ds is not None:
            probs = predict(model, val_ds)
            y_val = val_ds.labels
            row['val_loss'] = float(-np.log(probs[np.arange(len(y_val)), y_val] + 1e-12).mean())
            row.update({f'val_{k}': v for k, v in summary(y_val, probs).items()})
            if row['val_macro_f1'] > best['macro_f1']:
                best = {'macro_f1': row['val_macro_f1'], 'epoch': epoch,
                        'state': copy.deepcopy(model.state_dict()), 'probs': probs}
        row['seconds'] = time.time() - t0
        history.append(row)
        log(' '.join(f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}' for k, v in row.items()))
        if on_epoch is not None:
            on_epoch(epoch, row)

    if best['state'] is not None:
        model.load_state_dict(best['state'])
    return model, history, best


def run_cv(name, p, folds_to_run=None, tag=None, log=print, on_fold=None):
    """Stratified Group K-Fold na zbiorze treningowym. Zapisuje metryki foldów,
    historię uczenia i predykcje out-of-fold."""
    df = training_table()
    images = load_images(df.path)
    df['fold'] = make_folds(df)
    k = df.fold.max() + 1
    folds_to_run = range(k) if folds_to_run is None else folds_to_run
    results, oof = [], []
    for f in folds_to_run:
        tr, va = df.fold != f, df.fold == f
        log(f'--- {name} fold {f}: train {tr.sum()} val {va.sum()}')
        train_ds = MRIDataset(images[tr.to_numpy()], df.label[tr], train=True)
        val_ds = MRIDataset(images[va.to_numpy()], df.label[va])
        _, history, best = train_model(name, p, train_ds, val_ds, seed=SEED + f, log=log)
        res = {'fold': f, 'best_epoch': best['epoch'], 'history': history}
        res.update(summary(val_ds.labels, best['probs']))
        results.append(res)
        part = df[va][['path', 'label']].copy()
        part[[f'p_{c}' for c in CLASSES]] = best['probs']
        part['fold'] = f
        oof.append(part)
        if on_fold is not None:
            on_fold(f, res)
    if tag is not None:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_DIR / f'cv_{tag}.json', 'w') as fh:
            json.dump({'model': name, 'params': p, 'folds': results}, fh, indent=1)
        pd.concat(oof).to_csv(METRICS_DIR / f'oof_{tag}.csv', index=False)
    return results


def train_final(name, p, tag):
    """Model końcowy: cały zbiór treningowy, liczba epok ustalona z walidacji krzyżowej."""
    df = training_table()
    images = load_images(df.path)
    train_ds = MRIDataset(images, df.label, train=True)
    model, history, _ = train_model(name, p, train_ds)
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({'model': name, 'params': p, 'state_dict': model.state_dict()}, MODELS_DIR / f'{tag}.pth')
    with open(METRICS_DIR / f'final_{tag}_history.json', 'w') as fh:
        json.dump({'model': name, 'params': p, 'history': history}, fh, indent=1)


def load_trained(tag):
    ckpt = torch.load(MODELS_DIR / f'{tag}.pth', map_location=DEVICE, weights_only=True)
    model = build_model(ckpt['model'], ckpt['params']['dropout']).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['cv', 'final'])
    ap.add_argument('--model', required=True, choices=list(DEFAULTS))
    ap.add_argument('--params', help='plik JSON z hiperparametrami (nadpisuje domyślne)')
    ap.add_argument('--epochs', type=int, help='liczba epok (dla modelu końcowego)')
    ap.add_argument('--tag', help='nazwa plików wynikowych')
    args = ap.parse_args()

    p = dict(DEFAULTS[args.model])
    if args.params:
        with open(args.params) as fh:
            p.update(json.load(fh))
    if args.epochs:
        p['epochs'] = args.epochs
    tag = args.tag or args.model
    print('urządzenie:', DEVICE, 'parametry:', p, flush=True)
    if args.mode == 'cv':
        run_cv(args.model, p, tag=tag, log=lambda s: print(s, flush=True))
    else:
        train_final(args.model, p, tag)


if __name__ == '__main__':
    main()
