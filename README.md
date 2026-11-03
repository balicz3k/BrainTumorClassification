# Klasyfikacja Guzów Mózgu MRI

Projekt zaliczeniowy — Uczenie Maszynowe w Diagnostyce Medycznej (semestr 2).

## Cel

Klasyfikacja obrazów MRI mózgu na 4 klasy:
- **glioma** — glejakowatość
- **meningioma** — oponiak
- **notumor** — brak guza
- **pituitary** — guz przysadki

## Stack technologiczny

- Python 3.14
- PyTorch + torchvision (EfficientNetB0 — transfer learning)
- Pillow, NumPy, Matplotlib, scikit-learn
- Jupyter Notebook

## Dataset

**Brain Tumor MRI Dataset** (Masoud Nickparvar, Kaggle)
- 7023 obrazów MRI
- Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Pobranie danych

1. Zaloguj się na https://www.kaggle.com
2. Przejdź do Settings → API → Create New Token → pobierz `kaggle.json`
3. Umieść plik w `~/.kaggle/kaggle.json` i ustaw uprawnienia: `chmod 600 ~/.kaggle/kaggle.json`
4. Zainstaluj kaggle CLI: `pip install kaggle`
5. Pobierz dataset:

```bash
mkdir -p data/raw
cd data/raw
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip
```

Po rozpakowaniu struktura powinna wyglądać:
```
data/raw/
  Training/
    glioma/
    meningioma/
    notumor/
    pituitary/
  Testing/
    glioma/
    meningioma/
    notumor/
    pituitary/
```

## Uruchomienie

```bash
# Klonowanie repo
git clone https://github.com/balicz3k/BrainTumorClassification.git
cd BrainTumorClassification

# Środowisko wirtualne
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pobierz dane (patrz wyżej)

# Uruchom notebook EDA
jupyter notebook notebooks/01_eda.ipynb
```

## Struktura projektu

```
BrainTumorClassification/
  data/raw/          # dane surowe (w .gitignore — pobierz ręcznie)
  notebooks/
    01_eda.ipynb     # Eksploracyjna Analiza Danych
  src/               # kod modelu i treningu
  out/               # wykresy i wyniki (w .gitignore)
  requirements.txt
  README.md
```

## Podział danych

Planowany podział stratyfikowany: **70% trening / 15% walidacja / 15% test**
