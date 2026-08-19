# Grad-CAM Destekli Açıklanabilir Meyve Tazelik Sınıflandırma Sistemi

> İngilizce adı: *Fruit Freshness Classification and Explainable Image Analysis Using Grad-CAM*

Görüntü işleme ve derin öğrenme kullanarak meyvelerin **taze** mi yoksa **çürük** mü olduğunu sınıflandıran; verdiği kararları **Grad-CAM** ile görselleştirerek açıklayan bir Bilgisayarlı Görü projesidir.

---

## İçindekiler

- [Veri Seti (Dataset) Kurulumu](#veri-seti-dataset-kurulumu)
- [Proje Özeti](#proje-özeti)
- [Özellikler](#özellikler)
- [Veri Seti](#veri-seti)
- [Klasör Yapısı](#klasör-yapısı)
- [Kurulum](#kurulum)
- [Gereksinimler](#gereksinimler)
- [Kullanım](#kullanım)
- [Model Mimarisi](#model-mimarisi)
- [Performans Değerlendirme](#performans-değerlendirme)
- [Grad-CAM Açıklanabilirlik](#grad-cam-açıklanabilirlik)
- [Masaüstü Uygulama](#masaüstü-uygulama)
- [Akademik PDF Raporu](#akademik-pdf-raporu)
- [Sonuçlar ve Çıktılar](#sonuçlar-ve-çıktılar)
- [Sınırlamalar](#sınırlamalar)
- [Gelecek Çalışmalar](#gelecek-çalışmalar)
- [Kaynakça](#kaynakça)


---

## Veri Seti (Dataset) Kurulumu

Bu projenin veri seti boyutu (~8 GB) büyük olduğu için GitHub deposuna dahil edilmemiştir. Projeyi bilgisayarınızda çalıştırmadan önce aşağıdaki adımları uygulamalısınız:

1. **Veri Setini İndirin:** [Google Drive İndirme Bağlantısı](https://drive.google.com/file/d/1XH6kjIxgMUYA8SG-p6aOoVV44c5Ns7V1/view?usp=sharing) üzerinden ZIP dosyasını bilgisayarınıza indirin.
2. **ZIP'ten Çıkarın:** İndirdiğiniz `dataset.zip` dosyasının içeriğini ayıklayın.
3. Proje ana dizinine zipin içindeki iki dosyayı ekleyin(dataset ve temp_dataset, ikincisi geçici veri kümesi, incelenmesi adına eklenmiştir). Klasör yapısı, [Klasör Yapısı](#klasör-yapısı) başlığındaki mimariyle aynı olmalıdır.

## Proje Özeti

Bu sistem, MobileNetV2 tabanlı transfer öğrenme ile 6 sınıflı bir meyve tazelik sınıflandırıcısı eğitir, performansını ölçer ve modelin karar gerekçelerini Grad-CAM ısı haritalarıyla açıklar. Tüm akış (eğitim, değerlendirme, tahmin, açıklanabilirlik, masaüstü arayüz ve akademik PDF) modüler dosyalara bölünmüştür.

## Özellikler

- MobileNetV2 + Transfer Learning (iki aşamalı eğitim: özellik çıkarımı + ince ayar)
- Sınıf isimlerinin veri klasöründen **dinamik** tespiti (koda gömülü değil)
- Sınıf dengesizliğini telafi eden `class_weight`
- Accuracy, Precision, Recall, F1, Classification Report ve Confusion Matrix
- Son konvolüsyon katmanını **otomatik** tespit eden Grad-CAM
- PyQt5 masaüstü uygulaması (görsel seçme, tahmin, güven skoru, olasılık çubukları, Grad-CAM)
- Tek komutla üretilen akademik PDF raporu

## Veri Seti

Veri seti proje içindeki `dataset/` klasöründe bulunur ve önceden `train/` ile `test/` olarak ayrılmıştır.

**6 sınıf** (klasör adları, alfabetik):

| Sınıf | Eğitim | Test |
| --- | --- | --- |
| freshapples | 1853 | 435 |
| freshbanana | 1741 | 421 |
| freshoranges | 1626 | 428 |
| rottenapples | 2502 | 641 |
| rottenbanana | 2384 | 570 |
| rottenoranges | 1755 | 443 |
| **Toplam** | **11861** | **2938** |

**Mendeley ek verisi (CC BY 4.0):** [Fresh and Rotten Fruits Dataset](https://data.mendeley.com/datasets/bdd69gyhv8/1) (DOI: 10.17632/bdd69gyhv8.1) kaynağından **16 sınıf** (8 meyve × taze/çürük: elma, muz, portakal, üzüm, guava, hünnap, nar, çilek) `mendeley_aug_*` önekiyle `dataset/train|test` altına entegre edilmiştir. `outputs/ood_eval/` küçük holdout örneklemdir (kalibrasyon; eğitimle aynı sınıf klasör adları).

Notlar:
- Tüm görüntüler `.png` veya `.jpg` formatındadır; eğitimde hafif augmentation uygulanır.
- Sınıf isimlendirmesi tutarsızdır (`banana` tekil, diğerleri çoğul); bu yüzden sınıflar her yerde klasörlerden okunur.
- Hazır bir doğrulama (validation) kümesi yoktur; eğitim verisinden %20 ayrılır. `test/` yalnızca final değerlendirme için kullanılır.

## Klasör Yapısı

```
fruit-freshness-classification/
├── dataset/              # Veri seti (train/ ve test/)
├── models/              # Eğitilmiş model + class_indices.json
├── outputs/             # Grafikler, confusion matrix, Grad-CAM görselleri
├── reports/             # Classification report, metrikler, PDF
├── src/
│   ├── config.py        # Merkezi ayarlar
│   ├── data_loader.py   # Veri yükleme, dinamik sınıf, augmentation
│   ├── model_builder.py # MobileNetV2 model kurulumu
│   └── utils.py         # Grafik ve yardımcı fonksiyonlar
├── ui/
│   └── main_window.py   # Sekmeli profesyonel PyQt5 arayüzü
├── utils/
│   ├── predictor.py     # Tahmin servis katmanı
│   ├── gradcam.py       # Grad-CAM servis katmanı
│   └── image_utils.py   # Görüntü ve etiket yardımcıları
├── train.py             # Eğitim akışı
├── evaluate.py          # Değerlendirme akışı
├── predict.py           # Tek görsel tahmini
├── gradcam.py           # Grad-CAM üretimi
├── app.py               # PyQt5 masaüstü uygulaması
├── generate_report.py   # Akademik PDF üretimi
├── download_mendeley.py # Mendeley veri seti indirme
├── integrate_mendeley.py# Mendeley entegrasyonu + ood_eval
├── run_mendeley_pipeline.py # Tam otomatik pipeline
├── validate_mendeley.py # Hedef davranış doğrulama
├── requirements.txt
└── README.md
```

## Kurulum

```bash
# (Önerilir) Sanal ortam oluşturun
python -m venv venv
venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

## Gereksinimler

- Python 3.9
- TensorFlow 2.10.1, NumPy 1.23.5
- OpenCV, Pillow, Matplotlib, scikit-learn
- PyQt5 (masaüstü arayüz), reportlab (PDF)

Tüm sürümler `requirements.txt` içinde sabitlenmiştir.

## Kullanım

Aşağıdaki sırayla çalıştırın:

```bash
# 1) Modeli eğit (models/ ve outputs/ üretilir)
python train.py

# 2) Test setinde değerlendir (reports/ ve confusion matrix üretilir)
python evaluate.py

# 3) Tek görsel için tahmin (komut satırı)
python predict.py --image dataset/test/freshapples/ornek.png

# 4) Tek görsel için Grad-CAM
python gradcam.py --image dataset/test/rottenapples/ornek.png

# 5) Masaüstü uygulamasını başlat
python app.py

# 6) Akademik PDF raporunu üret
python generate_report.py

# (Opsiyonel) Mendeley veri güçlendirme + yeniden eğitim pipeline'ı
python run_mendeley_pipeline.py
python run_mendeley_pipeline.py --skip-train   # model hazırsa
python run_mendeley_pipeline.py --ood-only     # yalnızca OOD kalibrasyon
```

## Model Mimarisi

- **Temel model:** MobileNetV2 (ImageNet önceden eğitimli, `include_top=False`)
- **Üst katmanlar:** GlobalAveragePooling2D → Dropout(0.3) → Dense(softmax)
- **Görüntü boyutu:** 224×224, **batch:** 32
- **Aşama 1 (özellik çıkarımı):** taban donuk, LR=1e-3
- **Aşama 2 (ince ayar):** tabanın son katmanları açılır, LR=1e-5
- **Geri çağrımlar:** EarlyStopping (val_loss), ModelCheckpoint (val_accuracy)

Tüm parametreler `src/config.py` üzerinden değiştirilebilir.

## Performans Değerlendirme

`evaluate.py` test seti üzerinde şu metrikleri üretir:

- **Accuracy:** Genel doğruluk özeti.
- **Precision:** Yanlış pozitif maliyeti (çürüğü taze sanma riski).
- **Recall:** Kaçırılan gerçek vakalar (çürüğü yakalayamama).
- **F1 Score:** Dengesiz veride precision/recall dengesi.
- **Classification Report:** Sınıf bazında detaylı tablo.
- **Confusion Matrix:** Hangi sınıfların karıştığını gösteren görsel.

## Grad-CAM Açıklanabilirlik

Grad-CAM, modelin kararını verirken görüntünün hangi bölgelerine baktığını ısı haritası olarak gösterir. Bu sayede modelin gerçekten çürüklük lekelerine mi yoksa alakasız bölgelere mi odaklandığı doğrulanabilir. Akademik olarak projeye "açıklanabilirlik" (XAI) değeri katar.

## Masaüstü Uygulama

`app.py` ile başlatılan PyQt5 arayüzünde kullanıcı: görsel seçer, görseli görür, tahmin sonucunu ve güven skorunu görür, tüm sınıf olasılıklarını çubuklarla görür ve Grad-CAM ısı haritasını görüntüler. Model, donmayı önlemek için açılışta bir kez yüklenir.

## Akademik PDF Raporu

`generate_report.py`, üretilen tüm çıktıları (grafikler, confusion matrix, Grad-CAM, metrikler) tek bir akademik PDF dokümanında birleştirir. Doküman hem rapor hem sunum mantığı taşıyan 16 bölümden oluşur.

## Sonuçlar ve Çıktılar

- Eğitilmiş model: `models/fruit_model.h5`
- Doğruluk/kayıp grafikleri: `outputs/accuracy.png`, `outputs/loss.png`
- Confusion Matrix: `outputs/confusion_matrix.png`
- Grad-CAM görselleri: `outputs/gradcam/`
- Classification Report: `reports/classification_report.txt`
- Akademik PDF: `reports/akademik_rapor.pdf`

## Sınırlamalar

- Model yalnızca 3 meyve türü (elma, muz, portakal) için eğitilmiştir; diğer nesneler OOD katmanıyla "Bilinmeyen" veya "Sınırda" olarak işaretlenir.
- Guava ve nar gibi bazı yabancı meyveler embedding uzayında eğitim dağılımına yakın kalabilir (sınırlı OOD ayrımı).
- CPU üzerinde eğitim uzun sürebilir.

## Gelecek Çalışmalar

- Ek gerçek dünya kaynakları (Roboflow vb.) ile veri çeşitliliğinin artırılması.
- Daha güçlü mimariler (EfficientNet) ile karşılaştırma.
- Mobil/gömülü cihazlarda gerçek zamanlı çalışma.

## Kaynakça

- Sandler, M. ve ark. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.*
- Selvaraju, R. R. ve ark. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
- Mendeley Data (2020). *Fresh and Rotten Fruits Dataset.* DOI: [10.17632/bdd69gyhv8.1](https://doi.org/10.17632/bdd69gyhv8.1) (CC BY 4.0).
- TensorFlow ve Keras resmi dokümantasyonu.


