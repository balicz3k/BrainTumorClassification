"""Wczytywanie obrazów MRI, podział na foldy i augmentacja."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'raw'
CACHE_DIR = ROOT / 'data' / 'cache'
OUT_DIR = ROOT / 'out'
METRICS_DIR = OUT_DIR / 'metrics'
FIG_DIR = OUT_DIR / 'figures'
MODELS_DIR = ROOT / 'models'

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 224
SEED = 42


def list_images(split):
    """Lista plików z katalogu Training albo Testing (ścieżki względne do repo)."""
    rows = []
    for label, cls in enumerate(CLASSES):
        for p in sorted((DATA_DIR / split / cls).glob('*.jpg')):
            rows.append({'path': str(p.relative_to(ROOT)), 'split': split, 'cls': cls,
                         'label': label, 'aug': '-aug-' in p.name})
    return pd.DataFrame(rows)


def load_image(path, size=IMG_SIZE):
    """Obraz w skali szarości przeskalowany do size x size (uint8).

    Część plików jest zapisana jako RGB, ale kanały są (prawie) identyczne,
    więc trzymamy jeden kanał."""
    img = Image.open(path).convert('L').resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def load_images(paths, size=IMG_SIZE):
    """Wczytuje wszystkie obrazy do pamięci; wynik zapisuje w data/cache,
    żeby kolejne uruchomienia nie dekodowały JPG od nowa."""
    paths = list(paths)
    key = hashlib.md5('\n'.join(paths).encode()).hexdigest()[:10]
    cache = CACHE_DIR / f'img_{size}_{len(paths)}_{key}.npy'
    if cache.exists():
        return np.load(cache)
    images = np.stack([load_image(ROOT / p, size) for p in paths])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, images)
    return images


def load_split_table():
    """Tabela wszystkich obrazów z wynikami kontroli duplikatów (src/duplicates.py)."""
    path = METRICS_DIR / 'duplicates.csv'
    if not path.exists():
        raise FileNotFoundError('Brak out/metrics/duplicates.csv - uruchom najpierw: python -m src.duplicates')
    return pd.read_csv(path, dtype={'phash': str, 'md5': str})


def training_table():
    """Zbiór treningowy: katalog Training bez sztucznie augmentowanych kopii (Tr-aug-me)."""
    df = load_split_table()
    return df[(df.split == 'Training') & ~df.aug].reset_index(drop=True)


def test_table(dedup=False):
    """Zbiór testowy: oryginalny (1600 obrazów) albo oczyszczony z duplikatów treningu."""
    df = load_split_table()
    df = df[df.split == 'Testing']
    if dedup:
        df = df[df.test_dedup]
    return df.reset_index(drop=True)


def make_folds(df, k=5, seed=SEED):
    """Stratified Group K-Fold: proporcje klas zachowane w każdym foldzie,
    a obrazy z jednej grupy niemal-duplikatów zawsze w tym samym foldzie."""
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold = np.zeros(len(df), dtype=int)
    for i, (_, val_idx) in enumerate(sgkf.split(df, df.label, groups=df.group)):
        fold[val_idx] = i
    return fold


# Augmentacja dopasowana do MRI: małe obroty, przesunięcia i skalowanie oraz odbicie
# w poziomie (mózg jest w przybliżeniu symetryczny). Bez zmian jasności i kontrastu,
# bo intensywność sygnału niesie informację diagnostyczną.
TRAIN_AUG = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
])


class MRIDataset(Dataset):
    """Obrazy trzymane w pamięci jako uint8 (N, H, W); zwraca tensor 1 x H x W w [0, 1]."""

    def __init__(self, images, labels, train=False):
        self.images = images
        self.labels = np.asarray(labels)
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i]).float().div(255).unsqueeze(0)
        if self.train:
            x = TRAIN_AUG(x)
        return x, int(self.labels[i])
