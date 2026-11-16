"""Prosty interfejs do modelu: wgranie obrazu MRI -> klasa, prawdopodobieństwa i mapa Grad-CAM.

Uruchomienie: streamlit run app.py
Wymaga wytrenowanego modelu w katalogu models/ (python -m src.train final ...).
"""
import pandas as pd
import streamlit as st
import torch

from src.data import CLASSES, DATA_DIR, MODELS_DIR, load_image
from src.gradcam import grad_cam, overlay
from src.train import DEVICE, load_trained

NAMES_PL = {'glioma': 'glejak', 'meningioma': 'oponiak', 'notumor': 'brak guza', 'pituitary': 'guz przysadki'}

st.set_page_config(page_title='Klasyfikacja guzów mózgu MRI')
st.title('Klasyfikacja guzów mózgu na obrazach MRI')
st.warning('Narzędzie edukacyjne, wykonane w ramach projektu studenckiego. Nie jest wyrobem medycznym '
           'i nie może służyć do diagnozy. Model uczono na publicznym zbiorze z Kaggle, '
           'bez walidacji na danych z innego ośrodka.')

KNOWN = ['resnet50_tuned', 'resnet50', 'efficientnet_b0', 'cnn']  # modele zapisane przez src/train.py
tags = [t for t in KNOWN if (MODELS_DIR / f'{t}.pth').exists()]
if not tags:
    st.error('Brak modeli w katalogu models/. Najpierw wytrenuj model: python -m src.train final ...')
    st.stop()
tag = st.selectbox('Model', tags)


@st.cache_resource
def get_model(tag):
    return load_trained(tag)


uploaded = st.file_uploader('Obraz MRI (JPG lub PNG)', type=['jpg', 'jpeg', 'png'])
source = uploaded
# do pokazu: możliwość wyboru obrazu z katalogu Testing (jeśli dane są pobrane)
examples = sorted(str(p.relative_to(DATA_DIR)) for p in DATA_DIR.glob('Testing/*/*.jpg'))
if uploaded is None and examples:
    choice = st.selectbox('albo wybierz obraz z katalogu Testing', [''] + examples)
    if choice:
        source = DATA_DIR / choice

if source is not None:
    image = load_image(source)  # skala szarości, 224 x 224
    x = torch.tensor(image).float().div(255)[None, None].to(DEVICE)
    cam, pred, probs = grad_cam(get_model(tag), x)

    st.subheader(f'Predykcja: {NAMES_PL[CLASSES[pred]]} ({probs[pred]:.1%})')
    col1, col2 = st.columns(2)
    col1.image(image, caption='Obraz wejściowy (224 x 224)', width='stretch')
    col2.image(overlay(image, cam), caption='Grad-CAM dla przewidzianej klasy', width='stretch')

    table = pd.DataFrame({'klasa': [NAMES_PL[c] for c in CLASSES], 'prawdopodobieństwo': probs})
    st.bar_chart(table, x='klasa', y='prawdopodobieństwo')
    st.caption('Mapa Grad-CAM pokazuje obszary, które najbardziej wpłynęły na wynik sieci. '
               'Nie jest to segmentacja guza i nie dowodzi, że model rozumuje jak radiolog.')
