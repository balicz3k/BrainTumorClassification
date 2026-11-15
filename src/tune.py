"""Strojenie hiperparametrów biblioteką Optuna (próbnik TPE + przycinanie MedianPruner).

Cel: średnie macro-F1 na foldach walidacyjnych (zbiór testowy nie jest używany).
Żeby zmieścić się w czasie, każda próba używa 3 z 5 foldów i krótszego treningu.
Uruchomienie: python -m src.tune --model efficientnet_b0 --trials 20
"""
import argparse
import json

import numpy as np
import optuna

from src.data import METRICS_DIR, OUT_DIR
from src.train import DEFAULTS, run_cv

TUNE_FOLDS = [0, 1, 2]
TUNE_EPOCHS = 10
N_BLOCKS = {'efficientnet_b0': 9, 'resnet50': 5}


def make_objective(name):
    def objective(trial):
        p = dict(DEFAULTS[name])
        p['epochs'] = TUNE_EPOCHS
        p['lr'] = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
        p['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        p['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        p['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        if name in N_BLOCKS:
            p['n_unfreeze'] = trial.suggest_int('n_unfreeze', 1, N_BLOCKS[name])

        scores, best_epochs = [], []

        def on_fold(fold, res):
            scores.append(res['macro_f1'])
            best_epochs.append(res['best_epoch'])
            # po każdym foldzie raportujemy średnią; słabe próby są przerywane wcześnie
            trial.report(float(np.mean(scores)), step=len(scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        run_cv(name, p, folds_to_run=TUNE_FOLDS, log=lambda s: None, on_fold=on_fold)
        trial.set_user_attr('fold_scores', scores)
        trial.set_user_attr('best_epochs', best_epochs)
        return float(np.mean(scores))
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='efficientnet_b0')
    ap.add_argument('--trials', type=int, default=20)
    args = ap.parse_args()

    study = optuna.create_study(
        study_name=f'tune_{args.model}', direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        storage=f'sqlite:///{OUT_DIR / "optuna.db"}', load_if_exists=True)
    study.optimize(make_objective(args.model), n_trials=args.trials)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(METRICS_DIR / f'optuna_{args.model}.csv', index=False)
    best = dict(study.best_params)
    with open(METRICS_DIR / f'optuna_{args.model}_best.json', 'w') as fh:
        json.dump(best, fh, indent=1)
    print('najlepsze macro-F1:', study.best_value, 'parametry:', best)


if __name__ == '__main__':
    main()
