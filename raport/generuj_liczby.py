"""Generuje liczby (makra LaTeX-a) i tabele do raportu z plików w out/metrics.

Żadna liczba w raporcie nie jest wpisywana ręcznie - raport używa makr z liczby.tex
i tabel z katalogu tabele/. Uruchomienie z katalogu repozytorium:
    python raport/generuj_liczby.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import CLASSES, METRICS_DIR  # noqa: E402
from src.models import build_model, count_params  # noqa: E402
from src.tune import N_BLOCKS  # noqa: E402

OUT = ROOT / 'raport'
TAB = OUT / 'tabele'
TAB.mkdir(exist_ok=True)
BEST = sys.argv[1] if len(sys.argv) > 1 else 'resnet50'
TAGS = {'cnn': 'CNN od zera', 'efficientnet_b0': 'EfficientNetB0', 'resnet50': 'ResNet50',
        f'{BEST}_tuned': 'EfficientNetB0 (Optuna)' if BEST == 'efficientnet_b0' else 'ResNet50 (Optuna)'}
NAZWY = {'glioma': 'glejak', 'meningioma': 'oponiak', 'notumor': 'brak guza', 'pituitary': 'guz przysadki'}

macros = {}


def m(name, value):
    macros[name] = value


def pct(x, d=1):
    return f'{100 * x:.{d}f}\\,\\%'.replace('.', ',')


def num(x, d=3):
    return f'{x:.{d}f}'.replace('.', ',')


def ci(r, key, d=3):
    lo, hi = r['ci95'][key]
    return f'{num(lo, d)}--{num(hi, d)}'


def write_table(name, header, rows, align):
    lines = [f'\\begin{{tabular}}{{{align}}}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    for r in rows:
        lines.append('\\midrule' if r == 'midrule' else ' & '.join(str(x) for x in r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (TAB / f'{name}.tex').write_text('\n'.join(lines) + '\n', encoding='utf8')


# --- dane i duplikaty ---------------------------------------------------------
dup = pd.read_csv(METRICS_DIR / 'duplicates.csv', dtype={'phash': str, 'md5': str})
pairs = pd.read_csv(METRICS_DIR / 'duplicate_pairs.csv')
leak = pd.read_csv(METRICS_DIR / 'leak_vs_threshold.csv')
train = dup[dup.split == 'Training']
test = dup[dup.split == 'Testing']
m('nAll', len(dup))
m('nTrainDir', len(train))
m('nTestDir', len(test))
m('nAugTrain', int(train.aug.sum()))
m('nAugTest', int(test.aug.sum()))
m('nTrain', int((~train.aug).sum()))
m('nTestDedup', int(dup.test_dedup.sum()))
m('nDedupNotumor', int(((test.cls == 'notumor') & test.test_dedup).sum()))
md5_groups = dup.groupby('md5').size()
m('nMdGroups', int((md5_groups > 1).sum()))
m('nMdFiles', int(md5_groups[md5_groups > 1].sum()))
m('nPairs', len(pairs))
m('nPairsConflict', int((pairs.cls_a != pairs.cls_b).sum()))
m('nPairsConflictGM', int(((pairs.cls_a != pairs.cls_b) & pairs.cls_a.isin(['glioma', 'meningioma'])
                            & pairs.cls_b.isin(['glioma', 'meningioma'])).sum()))
for t in [0, 4]:
    row = leak[leak.threshold == t].iloc[0]
    word = {0: 'Zero', 4: 'Four'}[t]
    m(f'leak{word}', int(row.test_with_train_twin))
    m(f'leak{word}Notumor', int(row.notumor))
groups = train[~train.aug].groupby('group').size()
m('nGroups', len(groups))
m('nGroupsMax', int(groups.max()))
m('nGroupsSingle', int((groups == 1).sum()))

rows = []
for c in CLASSES:
    rows.append([c, int(((train.cls == c)).sum()), int(((train.cls == c) & train.aug).sum()),
                 int(((train.cls == c) & ~train.aug).sum()), int((test.cls == c).sum()),
                 int(((test.cls == c) & test.aug).sum()), int(((test.cls == c) & test.test_dedup).sum())])
rows.append(['razem'] + [sum(r[i] for r in rows) for i in range(1, 7)])
write_table('dane', ['klasa', 'Training', 'w tym aug', 'trening', 'Testing', 'w tym aug', 'test\\_dedup'],
            rows[:-1] + ['midrule', rows[-1]], 'lrrrrrr')

rows = [[int(r.threshold), int(r.test_with_train_twin)] + [int(r[c]) for c in CLASSES] for _, r in leak.iterrows()]
write_table('wyciek', ['próg pHash', 'razem'] + CLASSES, rows, 'rrrrrr')

eda = json.load(open(METRICS_DIR / 'eda_stats.json'))
m('nResolutions', eda['n_resolutions'])
m('fracFiveNotumor', pct(eda['frac_512']['notumor']))
m('fracFiveTumorMin', pct(min(eda['frac_512'][c] for c in ['glioma', 'meningioma', 'pituitary']), 0))
m('meanIntGlioma', num(eda['mean_intensity']['glioma'], 1))
m('meanIntNotumor', num(eda['mean_intensity']['notumor'], 1))
m('nTrainMeningioma', int(((train.cls == 'meningioma') & ~train.aug).sum()))

# --- walidacja krzyżowa --------------------------------------------------------
cv_rows = []
for tag, name in TAGS.items():
    cv = json.load(open(METRICS_DIR / f'cv_{tag}.json'))
    f = pd.DataFrame([{k: v for k, v in x.items() if k != 'history'} for x in cv['folds']])
    sec = np.mean([np.mean([h['seconds'] for h in x['history']]) for x in cv['folds']])
    key = tag.replace('_tuned', '')
    params = count_params(build_model(key)) / 1e6
    cv_rows.append([name, num(params, 1), f'{sec:.0f}', f'{f.best_epoch.median():.0f}/{cv["params"]["epochs"]}'] +
                   [f'{num(f[c].mean())} $\\pm$ {num(f[c].std())}' for c in
                    ['accuracy', 'macro_f1', 'macro_recall', 'macro_specificity', 'macro_auc']])
    short = {'cnn': 'Cnn', 'efficientnet_b0': 'Eff', 'resnet50': 'Res'}[key] + ('Tuned' if 'tuned' in tag else '')
    m(f'cvF{short}', num(f.macro_f1.mean()))
    m(f'cvFsd{short}', num(f.macro_f1.std()))
    m(f'cvAcc{short}', num(f.accuracy.mean()))
    m(f'cvSec{short}', f'{sec:.0f}')
    m(f'cvParams{short}', num(params, 1))
    m(f'cvBestEpoch{short}', f'{f.best_epoch.median():.0f}')
write_table('cv', ['model', 'mln par.', 's/ep.', 'najl. ep.', 'accuracy', 'macro-F1', 'macro czułość',
                   'macro swoistość', 'macro AUC'], cv_rows, 'lrrrccccc')

# --- Optuna ---------------------------------------------------------------------
trials = pd.read_csv(METRICS_DIR / f'optuna_{BEST}.csv')
best = json.load(open(METRICS_DIR / f'optuna_{BEST}_best.json'))
done = trials[trials.state == 'COMPLETE']
m('nTrials', int(trials.state.isin(['COMPLETE', 'PRUNED']).sum()))
m('nTrialsFail', int((trials.state == 'FAIL').sum()))
m('nTrialsComplete', len(done))
m('nTrialsPruned', int((trials.state == 'PRUNED').sum()))
m('optBest', num(done.value.max()))
m('optWorst', num(done.value.min()))
m('optLr', f'{best["lr"]:.2e}'.replace('e-0', '\\cdot 10^{-').replace('.', ',') + '}')
m('optWd', f'{best["weight_decay"]:.2e}'.replace('e-0', '\\cdot 10^{-').replace('.', ',') + '}')
m('optDropout', num(best['dropout'], 2))
m('optBatch', best['batch_size'])
m('optUnfreeze', best.get('n_unfreeze', '-'))
m('nBlocks', N_BLOCKS.get(BEST, '-'))

# --- test -------------------------------------------------------------------------
res = json.load(open(METRICS_DIR / 'test_metrics.json'))
rows = []
for tag, name in TAGS.items():
    for split, label in [('test', 'test'), ('test_dedup', 'test\\_dedup')]:
        r = res[tag][split]
        s = r['summary']
        rows.append([name if split == 'test' else '', label,
                     f"{num(s['accuracy'])} ({ci(r, 'accuracy')})",
                     f"{num(s['macro_f1'])} ({ci(r, 'macro_f1')})",
                     f"{num(s['macro_auc'])} ({ci(r, 'macro_auc')})",
                     num(s['macro_recall']), num(s['macro_specificity']), num(r['ece'])])
    short = {'cnn': 'Cnn', 'efficientnet_b0': 'Eff', 'resnet50': 'Res'}[tag.replace('_tuned', '')] + ('Tuned' if 'tuned' in tag else '')
    for split, sp in [('test', 'Test'), ('test_dedup', 'Dedup')]:
        r = res[tag][split]
        m(f'acc{sp}{short}', num(r['summary']['accuracy']))
        m(f'accCi{sp}{short}', ci(r, 'accuracy'))
        m(f'f{sp}{short}', num(r['summary']['macro_f1']))
        m(f'fCi{sp}{short}', ci(r, 'macro_f1'))
        m(f'auc{sp}{short}', num(r['summary']['macro_auc']))
        m(f'ece{sp}{short}', num(r['ece']))
write_table('test', ['model', 'zbiór', 'accuracy (95\\% CI)', 'macro-F1 (95\\% CI)', 'macro AUC (95\\% CI)',
                     'm. czuł.', 'm. swoist.', 'ECE'], rows, 'llcccrrr')

best_tag = f'{BEST}_tuned'
for split, label in [('test', 'test'), ('test_dedup', 'dedup')]:
    r = res[best_tag][split]
    rows = []
    for row in r['per_class']:
        c = row['klasa']
        rows.append([f'{c} ({NAZWY[c]})', row['n'], f"{num(row['czulosc'])} ({ci(r, f'czulosc_{c}')})",
                     f"{num(row['swoistosc'])} ({ci(r, f'swoistosc_{c}')})", num(row['precyzja']),
                     num(row['f1']), num(row['auc'])])
        cc = c.capitalize()
        sp = 'Test' if split == 'test' else 'Dedup'
        m(f'sens{sp}{cc}', num(row['czulosc']))
        m(f'sensCi{sp}{cc}', ci(r, f'czulosc_{c}'))
        m(f'spec{sp}{cc}', num(row['swoistosc']))
        m(f'specCi{sp}{cc}', ci(r, f'swoistosc_{c}'))
        m(f'auc{sp}{cc}', num(row['auc']))
    write_table(f'klasy_{label}', ['klasa', 'n', 'czułość (95\\% CI)', 'swoistość (95\\% CI)', 'precyzja', 'F1', 'AUC'],
                rows, 'lrccrrr')

preds = pd.read_csv(METRICS_DIR / f'test_preds_{best_tag}.csv')
err = preds[preds.pred != preds.label]
m('nErrTest', len(err))
m('nErrDedup', int(err.test_dedup.sum()))
pair_counts = (err.cls + ' $\\rightarrow$ ' + err.pred.map(dict(enumerate(CLASSES)))).value_counts()
write_table('bledy', ['prawda $\\rightarrow$ predykcja', 'liczba'], [[k, v] for k, v in pair_counts.items()], 'lr')
gm = int(((err.cls == 'glioma') & (err.pred == 1)).sum() + ((err.cls == 'meningioma') & (err.pred == 0)).sum())
m('nErrGM', gm)
m('fracErrGM', pct(gm / max(len(err), 1), 0))

# błędy a bliźniaki w treningu
d = preds.merge(dup[['path', 'min_dist_train', 'nearest_train']], on='path')
cls_of = dup.set_index('path').cls
d['twin_same'] = (d.min_dist_train <= 4) & (cls_of.loc[d.nearest_train].to_numpy() == d.cls)
d['err'] = d.pred != d.label
m('errRateTwin', pct(d[d.twin_same].err.mean()))
m('errRateNoTwin', pct(d[d.min_dist_train > 4].err.mean()))

# --- Grad-CAM ---------------------------------------------------------------------
cam = pd.read_csv(METRICS_DIR / 'gradcam_stats.csv')
g = cam.groupby('klasa').mean(numeric_only=True)
rows = [[c, pct(g.loc[c, 'udzial_tla_w_obrazie']), pct(g.loc[c, 'masa_na_tle']), pct(g.loc[c, 'masa_na_brzegu'])]
        for c in CLASSES]
rows.append(['wszystkie', pct(cam.udzial_tla_w_obrazie.mean()), pct(cam.masa_na_tle.mean()), pct(cam.masa_na_brzegu.mean())])
write_table('gradcam', ['klasa', 'tło w obrazie', 'masa mapy na tle', 'masa mapy na brzegu'], rows[:-1] + ['midrule', rows[-1]], 'lrrr')
m('camBg', pct(cam.masa_na_tle.mean()))
m('camBgArea', pct(cam.udzial_tla_w_obrazie.mean()))
m('camBorder', pct(cam.masa_na_brzegu.mean()))
m('camBgNotumor', pct(g.loc['notumor', 'masa_na_tle']))
m('camBgOk', pct(cam[cam.trafna].masa_na_tle.mean()))
m('camBgErr', pct(cam[~cam.trafna].masa_na_tle.mean()))

# --- błędy a źródło obrazu (rozmiar 512x512 = zbiór figshare) ------------------------
src = pd.read_csv(METRICS_DIR / 'bledy_zrodlo.csv', index_col=0)
m('gliomaOtherTrain', int(src.loc['glioma', 'trening: nie 512x512']))
m('gliomaOtherTest', int(src.loc['glioma', 'test: nie 512x512']))
m('gliomaOtherErr', int(src.loc['glioma', 'test: błędy wśród nie 512x512']))
m('gliomaFiveTest', int(src.loc['glioma', 'test: 512x512']))
m('gliomaFiveErr', int(src.loc['glioma', 'test: błędy wśród 512x512']))
m('gliomaOtherErrRate', pct(src.loc['glioma', 'test: błędy wśród nie 512x512'] / src.loc['glioma', 'test: nie 512x512']))
m('notumorOtherTrain', int(src.loc['notumor', 'trening: nie 512x512']))
m('meningiomaOtherTrain', int(src.loc['meningioma', 'trening: nie 512x512']))
m('nErrGliomaNotumor', int(((err.cls == 'glioma') & (err.pred == 2)).sum()))
m('nErrGliomaMeningioma', int(((err.cls == 'glioma') & (err.pred == 1)).sum()))
m('nErrGlioma', int((err.cls == 'glioma').sum()))
m('camBgTumor', pct(cam[cam.klasa != 'notumor'].masa_na_tle.mean()))
m('camBorderNotumor', pct(g.loc['notumor', 'masa_na_brzegu']))
m('camBorderTumor', pct(cam[cam.klasa != 'notumor'].masa_na_brzegu.mean()))
m('borderArea', pct(1 - (180 / 224) ** 2, 0))

# --- Optuna: ważność hiperparametrów (zapisana w notatniku 04) -------------------------
imp = json.load(open(METRICS_DIR / f'optuna_{BEST}_importance.json'))
m('impLr', num(imp['lr'], 2))
m('impDropout', num(imp['dropout'], 2))
m('impWd', num(imp['weight_decay'], 2))

with open(OUT / 'liczby.tex', 'w', encoding='utf8') as fh:
    fh.write('% Plik generowany przez raport/generuj_liczby.py - nie edytować ręcznie.\n')
    for k, v in macros.items():
        fh.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print('zapisano', len(macros), 'makr i tabele w', TAB)
