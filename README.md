# Klasyfikacja guzów mózgu na obrazach MRI

Projekt z przedmiotu *Uczenie Maszynowe w Diagnostyce Medycznej* (Politechnika Krakowska,
studia magisterskie, semestr 2).

Klasyfikacja przekrojów MRI mózgu na 4 klasy: glejak (`glioma`), oponiak (`meningioma`),
brak guza (`notumor`), guz przysadki (`pituitary`). Porównujemy trzy modele (CNN od zera,
EfficientNetB0, ResNet50), stroimy najlepszy Optuną, oceniamy go metrykami klinicznymi
z przedziałami ufności i sprawdzamy Grad-CAM-em, na co patrzy.

Raport: [`raport/raport.pdf`](raport/raport.pdf). Materiał do obrony: [`docs/NAUKA.md`](docs/NAUKA.md).

## Dane

*Brain Tumor MRI Dataset* (Masoud Nickparvar), https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Pobrana wersja ma 7200 obrazów JPG: `Training` po 1400 i `Testing` po 400 na klasę.
Klasę `meningioma` autor zbioru dopełnił przekształconymi kopiami (`Tr-aug-me_*`, `Te-aug-me_*`),
a starsze opisy zbioru podają 7023 obrazy (1321/1339/1595/1457 w treningu). Szczegóły i kontrola
duplikatów: notatnik `01_eda.ipynb`.

Pobranie (wymaga tokenu API Kaggle w `~/.kaggle/kaggle.json`):

```bash
mkdir -p data/raw && cd data/raw
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip
```

Oczekiwana struktura: `data/raw/{Training,Testing}/{glioma,meningioma,notumor,pituitary}/*.jpg`.
Obrazy nie są w repozytorium.

## Środowisko

Python 3.12, PyTorch 2.6 z CUDA 12.4 (testowane na RTX 4060 8 GB, Kubuntu).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

## Uruchomienie (kolejno)

```bash
# 1. duplikaty i wyciek (tworzy out/metrics/duplicates.csv, potrzebny do treningu)
python -m src.duplicates

# 2. walidacja krzyżowa trzech modeli (ok. 50 min na RTX 4060)
scripts/run_cv.sh

# 3. strojenie, pełne CV po strojeniu, modele końcowe i jednorazowa ewaluacja na teście
scripts/run_final.sh resnet50

# 4. notatniki z wynikami (wczytują pliki z out/metrics)
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 0*.ipynb

# 5. raport
python raport/generuj_liczby.py resnet50 && cd raport && pdflatex raport.tex && pdflatex raport.tex

# GUI
streamlit run app.py
```

## Struktura

```
src/
  data.py        wczytywanie obrazów, podział StratifiedGroupKFold, augmentacja
  duplicates.py  MD5 i pHash, grupy niemal-duplikatów, oczyszczony test
  models.py      SimpleCNN, EfficientNetB0, ResNet50
  train.py       pętla treningowa, walidacja krzyżowa, model końcowy
  tune.py        Optuna (TPE + MedianPruner)
  metrics.py     czułość, swoistość, F1, ROC AUC, bootstrap CI, ECE
  evaluate.py    jednorazowa ewaluacja na teście
  gradcam.py     Grad-CAM na hookach
notebooks/
  01_eda.ipynb                     EDA, jakość danych, duplikaty
  02_baseline_cnn.ipynb            CNN od zera (wyniki CV)
  03_transfer_learning.ipynb       EfficientNetB0 i ResNet50, porównanie modeli
  04_tuning_optuna.ipynb           strojenie hiperparametrów
  05_ewaluacja_wyjasnialnosc.ipynb test, CI, ROC, kalibracja, Grad-CAM, analiza błędów
scripts/         uruchamianie długich treningów w tle
app.py           GUI (Streamlit)
out/metrics/     wyniki liczbowe (JSON, CSV) - źródło wszystkich liczb w raporcie
out/figures/     wykresy
raport/          raport LaTeX + PDF
docs/NAUKA.md    jak działa każdy element, pytania do obrony
models/          wytrenowane modele (poza gitem, odtwarzalne skryptami)
```
