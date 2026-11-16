# Jak działa projekt - materiał do nauki i obrony

Plik tłumaczy każdy element projektu w kolejności, w jakiej powstawał, i zbiera pytania,
które może zadać prowadzący. Najważniejszy wynik projektu to wykrycie wycieku
i uczenia się skrótu przez model (pytania 4, 12, 16). Liczby pochodzą z `out/metrics/` (te same co w raporcie).

## 1. Dane i ich problemy

**Zbiór.** 7200 przekrojów MRI z Kaggle, 4 klasy. Pobrana wersja jest sztucznie zbalansowana
(1400/400 na klasę): klasę `meningioma` dopełniono kopiami `*-aug-*` (100 w treningu, 103 w teście).
Kopie to odbite lub obrócone oryginały - nie są nowymi badaniami, więc je odrzucamy.

**Źródła.** Zbiór łączy kilka publicznych kolekcji (figshare, SARTAJ, Br35H). Widać to w danych:
obrazy z guzami to prawie zawsze 512x512 i T1 po kontraście, a obrazy `notumor` mają różne
rozmiary i sekwencje. Model może więc nauczyć się rozpoznawać *źródło* zamiast *guza*.

**Jasność.** Średnia jasność obrazów `glioma` jest wyraźnie niższa niż `notumor` (33 vs 62 na
skali 0-255). To efekt akwizycji, nie guza. Dlatego nie robimy augmentacji jasności/kontrastu
(intensywność w MRI niesie informację) i sprawdzamy Grad-CAM-em, gdzie patrzy model.

**Duplikaty (`src/duplicates.py`).**
- MD5 pliku - identyczne pliki (153 grupy).
- pHash (perceptual hash): obraz zmniejszony do 32x32, transformata DCT, z lewego górnego rogu
  8x8 współczynników bierzemy znak względem mediany -> 64 bity. Podobne obrazy mają podobne
  bity; odległość Hamminga = liczba różnych bitów.
- Próg 4 bity dobrany przez obejrzenie par: przy 0-2 to te same obrazy, przy 4 ten sam albo
  sąsiedni przekrój, przy 6+ coraz częściej różne osoby.
- Wynik: 714 z 1600 obrazów testowych ma bliźniaka w treningu (353 z 400 `notumor`!).
- Oczyszczony test `test_dedup` = test bez tych obrazów i bez `aug` (806 obrazów).
- Grupy niemal-duplikatów w treningu (spójne składowe grafu „odległość <= 4”) trafiają zawsze
  do jednego foldu walidacji (StratifiedGroupKFold).

## 2. Walidacja

- **Test** = katalog `Testing`, użyty raz na końcu (`src/evaluate.py`), do niczego nie wybieramy
  na jego podstawie.
- **Stratified Group K-Fold, K = 5** na 5500 obrazach treningowych: każdy fold ma te same
  proporcje klas (stratyfikacja) i nie rozdziela grup duplikatów (grupy). To przybliżenie
  podziału per pacjent (LOGO z wykładu 4), skoro zbiór nie ma identyfikatora pacjenta.
- W każdym foldzie zapamiętujemy najlepszą epokę wg macro-F1 na walidacji (zamiast ostatniej).
- Model końcowy: cały trening, ta sama liczba epok co w CV (harmonogram kosinusowy kończy się
  małym LR, więc ostatnia epoka jest zwykle blisko najlepszej - sprawdzone w CV).

## 3. Modele (`src/models.py`)

| model | po co | szczegóły |
|---|---|---|
| CNN od zera | punkt odniesienia | 4 bloki (2x Conv-BN-ReLU + MaxPool), 1 kanał, ok. 1,2 mln parametrów |
| EfficientNetB0 | mały i dokładny | ImageNet, 4 mln parametrów, odmrażane ostatnie bloki |
| ResNet50 | duży klasyk | ImageNet, 23,5 mln parametrów, odmrażane `layer3` i `layer4` (po strojeniu: `layer2`-`layer4`) |

**Transfer learning.** Wagi z ImageNet dają gotowe filtry krawędzi i tekstur. Najpierw 2 epoki
uczymy tylko nową głowicę (szkielet zamrożony), potem odmrażamy ostatnie bloki z LR 10 razy
mniejszym, żeby nie zniszczyć wag.

**Wejście.** Obraz szary 224x224. Dla sieci z ImageNet kopiujemy go na 3 kanały i normalizujemy
statystykami ImageNet - robi to sam model w `forward`, więc GUI i Grad-CAM podają ten sam tensor.

**Augmentacja.** Odbicie w poziomie (mózg jest w przybliżeniu symetryczny, a guz może być po
każdej stronie), obrót do 10 stopni, przesunięcie do 5%, skala 0,9-1,1 (różne ułożenie głowy
i kadrowanie). Bez jasności i kontrastu.

**Trening (`src/train.py`).** AdamW (Adam z poprawnym weight decay), entropia krzyżowa,
harmonogram kosinusowy, mieszana precyzja bf16 (szybciej i mniej pamięci na RTX 4060).

## 4. Strojenie (`src/tune.py`)

Optuna, próbnik TPE: dzieli dotychczasowe próby na lepsze i gorsze, modeluje rozkład
hiperparametrów w obu grupach i proponuje punkty, gdzie stosunek „dobre/złe” jest największy.
MedianPruner przerywa próbę, jeśli po kolejnym foldzie jest poniżej mediany innych prób.
Cel: średnie macro-F1 na foldach 0-2, 10 epok. Przestrzeń: LR, weight decay, dropout,
batch size, liczba odmrożonych bloków.

Ryzyko (wykład 4): przy wielu próbach wybieramy konfigurację, która „miała szczęście” na tych
foldach - wynik walidacji po strojeniu jest optymistyczny. Ochrona: test, którego strojenie
nie widzi, i porównanie z wynikiem przed strojeniem.

## 5. Metryki (`src/metrics.py`)

- **Czułość** klasy k = TP / (TP + FN): ile obrazów klasy k model wykrył.
- **Swoistość** klasy k (one-vs-rest) = TN / (TN + FP): ile obrazów *innych* klas model
  poprawnie nie nazwał klasą k.
- **Precyzja** = TP / (TP + FP), **F1** = średnia harmoniczna precyzji i czułości;
  **macro-F1** = zwykła średnia F1 z 4 klas (każda klasa waży tyle samo).
- **ROC AUC** (one-vs-rest): pole pod krzywą czułość vs 1-swoistość dla wszystkich progów;
  = prawdopodobieństwo, że losowy obraz klasy k dostanie wyższe p(k) niż losowy obraz innej klasy.
- **Bootstrap CI 95%**: 1000 razy losujemy ze zwracaniem zbiór testowy tej samej wielkości,
  liczymy metrykę, bierzemy 2,5 i 97,5 percentyl.
- **ECE** (kalibracja): dzielimy predykcje na 10 przedziałów pewności i liczymy ważoną średnią
  |trafność - średnia pewność|.

## 6. Grad-CAM (`src/gradcam.py`)

1. Hook zapamiętuje aktywacje A (C x h x w) ostatniej warstwy konwolucyjnej i ich gradient.
2. Liczymy gradient wyniku (logitu) wybranej klasy po A.
3. Waga kanału = średni gradient w tym kanale.
4. Mapa = ReLU(suma po kanałach waga * A), skalowana do 224x224 i do [0, 1].

Ograniczenia: niska rozdzielczość (7x7 dla EfficientNetB0), pokazuje korelację, nie przyczynę,
i łatwo ją nadinterpretować. Dlatego liczymy też ilościowo, jaka część mapy leży na tle
i na brzegu obrazu, oraz porównujemy z mapą saliency.

## 7. Pytania prowadzącego (z odpowiedziami)

1. **Dlaczego czułość jest ważniejsza od accuracy?**
   Accuracy liczy wszystkie błędy tak samo i przy niezbalansowanych klasach może być wysokie,
   choć model przeocza chorych. W diagnostyce guza przeoczenie (FN) jest najgroźniejsze -
   pacjent nie dostaje leczenia. Czułość mówi wprost, jaki odsetek chorych wykrywamy.

2. **Co to jest swoistość per klasa w problemie 4-klasowym?**
   Traktujemy klasę k jako „pozytywną”, resztę jako „negatywną” (one-vs-rest).
   Swoistość = odsetek obrazów spoza klasy k, których model nie przypisał do k. Dla `notumor`
   swoistość oznacza, jak rzadko pacjent z guzem dostaje wynik „brak guza” - to klinicznie
   najważniejsza liczba (fałszywe uspokojenie).

3. **Po co Stratified K-Fold, a nie zwykły podział?**
   Jeden podział train/val daje wynik zależny od losowania; 5 foldów daje średnią i odchylenie.
   Stratyfikacja pilnuje tych samych proporcji klas w każdym foldzie, więc metryki są stabilniejsze.
   Grupowanie dodatkowo nie rozdziela duplikatów między trening i walidację.

4. **Co to jest wyciek danych w medycynie i jak go sprawdziłeś?**
   Wyciek = informacja z testu dostępna przy uczeniu, np. przekroje tego samego pacjenta
   w treningu i teście; model rozpoznaje pacjenta, nie chorobę, i wynik jest zawyżony.
   Zbiór nie ma ID pacjenta, więc szukałem duplikatów MD5 i pHash; znalazłem 714 testowych
   obrazów z bliźniakiem w treningu, zrobiłem oczyszczony test i raportuję oba wyniki.

5. **Dlaczego transfer learning?**
   5500 obrazów to mało dla głębokiej sieci uczonej od zera. Wagi z ImageNet dają ogólne
   filtry (krawędzie, tekstury), więc uczymy mniej parametrów i szybciej, z lepszym wynikiem -
   co widać w porównaniu z CNN od zera.

6. **Dlaczego bez augmentacji jasności i kontrastu?**
   W MRI intensywność sygnału ma znaczenie (np. wzmocnienie kontrastowe guza). Zmiana jasności
   tworzy obrazy, które nie odpowiadają realnym sekwencjom. Dodatkowo jasność różni się między
   klasami z powodu źródeł danych - chcieliśmy ten problem pokazać, a nie zamaskować.

7. **Dlaczego odbicie w poziomie jest w porządku?**
   Mózg jest w przybliżeniu symetryczny lewo-prawo, a guz może wystąpić po każdej stronie,
   więc odbity obraz to wciąż realistyczny przypadek tej samej klasy. Odbicia w pionie nie
   stosujemy (mózg „do góry nogami” nie występuje).

8. **Co pokazuje Grad-CAM i jakie ma ograniczenia?**
   Obszary ostatniej mapy cech, które najbardziej podniosły wynik danej klasy. Ograniczenia:
   niska rozdzielczość, pokazuje korelację, a nie rozumowanie; różne metody (saliency, LIME)
   dają inne mapy; ładna mapa nie dowodzi poprawności (ryzyko nadinterpretacji, wykład 5).

9. **Jak działa Optuna?**
   Funkcja celu zwraca wynik walidacji dla zaproponowanych hiperparametrów. TPE uczy się z
   poprzednich prób, które obszary przestrzeni są obiecujące; pruner przerywa słabe próby
   po pierwszych foldach. Wynik: najlepsze parametry i ważność hiperparametrów.

10. **Jak czytać krzywą ROC?**
    Każdy punkt to inny próg na p(klasa). Oś X: odsetek fałszywych alarmów (1 - swoistość),
    oś Y: czułość. Im bliżej lewego górnego rogu, tym lepiej; przekątna = losowanie.
    AUC = 1 to idealne rozdzielenie, 0,5 = losowo.

11. **Co to bootstrap CI i po co?**
    Jeden wynik na teście zależy od tego, które obrazy w nim są. Losując test ze zwracaniem
    1000 razy, widzimy rozrzut metryki. Przedział 95% mówi, w jakim zakresie wynik mógłby
    wyjść na podobnym zbiorze. Na małym oczyszczonym teście (47 obrazów `notumor`) przedziały
    są szerokie - to uczciwa informacja o niepewności.

12. **Dlaczego wynik na oczyszczonym teście jest inny niż na oryginalnym?**
    Część oryginalnego testu to kopie obrazów treningowych - model je „pamięta”. Oczyszczony
    test jest bliższy sytuacji nowego pacjenta, ale wciąż pochodzi z tych samych źródeł,
    więc nie zastępuje walidacji zewnętrznej.

13. **Dlaczego model myli glejaka z oponiakiem?**
    Oba to guzy wewnątrzczaszkowe widoczne jako zmiany o podobnej intensywności po kontraście;
    rozróżnienie zależy od położenia (opony vs miąższ) i kształtu, a na pojedynczym przekroju
    2D bywa trudne. Dodatkowo w danych są te same obrazy z obiema etykietami (szum etykiet).

14. **Czy ten model można użyć w szpitalu?**
    Nie. Brak walidacji zewnętrznej (inny szpital, skaner), pojedyncze przekroje 2D zamiast
    całego badania 3D, brak ID pacjenta, sztucznie zbalansowany zbiór i różne źródła klas.
    Mógłby być co najwyżej punktem wyjścia do narzędzia wspierającego triage, po badaniach
    prospektywnych i certyfikacji jako wyrób medyczny.

15. **Jakie są aspekty etyczne i prawne (RODO)?**
    Dane medyczne to dane szczególnej kategorii (art. 9 RODO) - wymagają podstawy prawnej
    i anonimizacji. Pacjent ma prawo do informacji o zautomatyzowanym przetwarzaniu (art. 22),
    więc potrzebna jest wyjaśnialność. Odpowiedzialność za decyzję ponosi lekarz; model
    może wspierać, nie decydować. Ukryte uprzedzenia w danych (np. jedno źródło na klasę)
    nie są widoczne w samych metrykach.

16. **(Bardzo prawdopodobne) Dlaczego wynik na teście jest niższy niż w walidacji krzyżowej?**
    W teście są glejaki z innego źródła (inne rozmiary, sekwencje T2/FLAIR), których w treningu
    nie było wcale. Model nauczył się skrótu „glejak = obraz T1 po kontraście z figshare”
    (shortcut learning). Walidacja krzyżowa na treningu nie mogła tego wykryć, bo każdy fold
    miał tylko glejaki z figshare. Wykryła to analiza błędów: 66 z 76 takich glejaków jest
    błędnych, a wśród glejaków 512x512 tylko 1 z 324.

17. **Jak można by to naprawić?**
    Dodać do treningu glejaki z wielu źródeł i sekwencji, normalizować intensywność w obrębie
    mózgu, wycinać sam mózg (bez ramek i znaczników), walidować grupami według źródła
    (leave-one-source-out) i sprawdzić model na zewnętrznym zbiorze (np. BraTS).

## 8. Wyniki - skrót

Źródło: `out/metrics/` (te same pliki, z których raport generuje liczby).

| model | macro-F1 CV (5 foldów) | accuracy test | accuracy test_dedup | macro-F1 test_dedup |
|---|---|---|---|---|
| CNN od zera | 0,955 ± 0,006 | 0,919 | 0,872 | 0,822 |
| EfficientNetB0 | 0,981 ± 0,006 | 0,953 | 0,914 | 0,874 |
| ResNet50 | 0,984 ± 0,005 | 0,949 | 0,906 | 0,866 |
| ResNet50 po Optunie | 0,987 ± 0,006 | 0,957 | 0,917 | 0,886 |

Najważniejsze do zapamiętania:
- **Wyciek:** 714 z 1600 obrazów testowych ma bliźniaka w treningu. Na obrazach z bliźniakiem
  tej samej klasy model nie myli się wcale, bez bliźniaka - w 7,6% przypadków.
- **Glejaki z innego źródła:** w treningu wszystkie glejaki mają 512x512 (figshare). W teście
  76 glejaków ma inny rozmiar i model myli 66 z nich (42 błędy glejak -> oponiak i 25 glejak ->
  brak guza na całym teście). Czułość dla glejaka na test_dedup to 0,79, a dla pozostałych klas ok. 1.
- **Optuna:** najlepsze LR 2,2e-3, weight decay 1,5e-3, dropout 0,12, batch 16, 3 odmrożone bloki.
  Najważniejszy jest LR. Poprawa w CV mieści się w odchyleniu.
- **Grad-CAM:** dla klas z guzem mapa skupia się na zmianie; dla `notumor` jest rozlana,
  z ok. 37% masy na tle i 24% na brzegu kadru.
