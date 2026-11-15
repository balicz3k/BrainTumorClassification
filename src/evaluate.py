"""Jednorazowa ewaluacja modeli końcowych na zbiorze testowym.

Zapisuje predykcje (out/metrics/test_preds_<tag>.csv) i metryki z przedziałami ufności
(out/metrics/test_metrics.json) dla oryginalnego i oczyszczonego testu.
Uruchomienie: python -m src.evaluate cnn efficientnet_b0_tuned resnet50
"""
import argparse
import json

from src.data import CLASSES, METRICS_DIR, MRIDataset, load_images, test_table
from src.metrics import bootstrap_ci, expected_calibration_error, per_class_table, summary
from src.train import load_trained, predict


def evaluate(y, probs):
    return {'summary': summary(y, probs),
            'per_class': per_class_table(y, probs),
            'ci95': bootstrap_ci(y, probs, n_boot=1000),
            'ece': expected_calibration_error(y, probs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+')
    args = ap.parse_args()

    test = test_table()
    images = load_images(test.path)
    dataset = MRIDataset(images, test.label)
    results = {}
    for tag in args.tags:
        probs = predict(load_trained(tag), dataset)
        preds = test[['path', 'cls', 'label', 'test_dedup', 'aug']].copy()
        preds[[f'p_{c}' for c in CLASSES]] = probs
        preds['pred'] = probs.argmax(1)
        preds.to_csv(METRICS_DIR / f'test_preds_{tag}.csv', index=False)

        dedup = test.test_dedup.to_numpy()
        results[tag] = {'test': evaluate(test.label.to_numpy(), probs),
                        'test_dedup': evaluate(test.label.to_numpy()[dedup], probs[dedup])}
        s1, s2 = results[tag]['test']['summary'], results[tag]['test_dedup']['summary']
        print(f"{tag}: accuracy test {s1['accuracy']:.4f}, test_dedup {s2['accuracy']:.4f}")

    with open(METRICS_DIR / 'test_metrics.json', 'w') as fh:
        json.dump(results, fh, indent=1, default=float)


if __name__ == '__main__':
    main()
