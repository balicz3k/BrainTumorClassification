"""Kontrola duplikatów i wycieku między Training i Testing.

Zbiór nie ma identyfikatora pacjenta, więc zamiast podziału per pacjent szukamy
obrazów identycznych (MD5 pliku) i niemal identycznych (perceptual hash).
Uruchomienie: python -m src.duplicates
"""
import hashlib

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse.csgraph import connected_components

from src.data import METRICS_DIR, ROOT, list_images

# Maksymalna odległość Hamminga między 64-bitowymi pHash, przy której dwa obrazy
# uznajemy za ten sam przekrój (albo sąsiedni przekrój tego samego badania).
# Próg dobrany na podstawie obejrzanych par w notatniku 01_eda.
PHASH_THRESHOLD = 4


def phash_bits(path):
    img = Image.open(ROOT / path).convert('L')
    return imagehash.phash(img).hash.flatten()


def hamming_matrix(bits):
    """Macierz odległości Hamminga między wszystkimi parami (7200 x 7200 mieści się w RAM)."""
    bits = np.asarray(bits, dtype=bool)
    return (bits[:, None, :] != bits[None, :, :]).sum(axis=2)


def find_duplicates(threshold=PHASH_THRESHOLD):
    df = pd.concat([list_images('Training'), list_images('Testing')], ignore_index=True)
    df['md5'] = [hashlib.md5((ROOT / p).read_bytes()).hexdigest() for p in df.path]
    bits = [phash_bits(p) for p in df.path]
    df['phash'] = [''.join('1' if b else '0' for b in x) for x in bits]
    dist = hamming_matrix(bits)
    np.fill_diagonal(dist, 999)

    is_train = (df.split == 'Training').to_numpy()
    is_test = ~is_train

    # najbliższy obraz treningowy dla każdego obrazu
    d_train = dist[:, is_train]
    train_paths = df.path[is_train].to_numpy()
    df['min_dist_train'] = d_train.min(axis=1)
    df['nearest_train'] = train_paths[d_train.argmin(axis=1)]

    # grupy niemal-duplikatów w treningu (spójne składowe grafu "odległość <= próg");
    # używane w StratifiedGroupKFold, żeby kopie nie trafiały do różnych foldów
    train_main = is_train & ~df.aug.to_numpy()
    adj = (dist <= threshold) & train_main[:, None] & train_main[None, :]
    _, comp = connected_components(adj, directed=False)
    df['group'] = np.where(train_main, comp, -1)

    # oczyszczony test: bez obrazów, które mają bliźniaka w treningu, i bez sztucznych kopii
    df['test_dedup'] = is_test & (df.min_dist_train > threshold) & ~df.aug.to_numpy()

    # wszystkie pary niemal-duplikatów (do analizy w EDA)
    i, j = np.where(np.triu(dist <= threshold, 1))
    pairs = pd.DataFrame({'path_a': df.path.to_numpy()[i], 'path_b': df.path.to_numpy()[j],
                          'split_a': df.split.to_numpy()[i], 'split_b': df.split.to_numpy()[j],
                          'cls_a': df.cls.to_numpy()[i], 'cls_b': df.cls.to_numpy()[j],
                          'dist': dist[i, j], 'same_md5': df.md5.to_numpy()[i] == df.md5.to_numpy()[j]})
    return df, pairs, dist


def main():
    df, pairs, dist = find_duplicates()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(METRICS_DIR / 'duplicates.csv', index=False)
    pairs.to_csv(METRICS_DIR / 'duplicate_pairs.csv', index=False)

    # wrażliwość wyniku na wybór progu
    is_train = (df.split == 'Training').to_numpy()
    is_test = ~is_train
    rows = []
    for t in [0, 2, 4, 6, 8]:
        leak = is_test & (df.min_dist_train <= t).to_numpy()
        row = {'threshold': t, 'test_with_train_twin': int(leak.sum())}
        for cls in sorted(df.cls.unique()):
            row[cls] = int((leak & (df.cls == cls).to_numpy()).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(METRICS_DIR / 'leak_vs_threshold.csv', index=False)
    print('test_dedup:', int(df.test_dedup.sum()), 'z', int(is_test.sum()))


if __name__ == '__main__':
    main()
