"""Merkezi yapilandirma modulu.

Projedeki tum sihirli sayilar, yollar ve egitim ayarlari burada
toplanir. Boylece train.py, evaluate.py, predict.py, gradcam.py ve
app.py ayni ayarlari paylasir; tek yerden degisiklik yapilir.
"""

from pathlib import Path

# ----------------------------------------------------------------
# KLASOR YOLLARI
# ----------------------------------------------------------------
# Proje kok dizini (bu dosya src/ icinde oldugu icin iki ust dizin).
PROJE_KOK = Path(__file__).resolve().parent.parent

# Veri seti yollari (gercek yapi: train/ ve test/ onceden bolunmus).
VERI_SETI_DIZIN = PROJE_KOK / "dataset"
EGITIM_DIZIN = VERI_SETI_DIZIN / "train"
TEST_DIZIN = VERI_SETI_DIZIN / "test"

# Cikti klasorleri.
MODEL_DIZIN = PROJE_KOK / "models"
CIKTI_DIZIN = PROJE_KOK / "outputs"
RAPOR_DIZIN = PROJE_KOK / "reports"
GRADCAM_DIZIN = CIKTI_DIZIN / "gradcam"

# Kayit dosyalari.
# TensorFlow 2.10 .keras formatini tam desteklemedigi icin .h5 kullanilir.
MODEL_DOSYA = MODEL_DIZIN / "fruit_model.h5"
SINIF_INDEKS_DOSYA = MODEL_DIZIN / "class_indices.json"
EGITIM_GECMIS_DOSYA = CIKTI_DIZIN / "egitim_gecmisi.json"
# OOD (bilinmeyen tur) istatistikleri.
OOD_ISTATISTIK_DOSYA = MODEL_DIZIN / "ood_stats.json"

# ----------------------------------------------------------------
# GORUNTU VE EGITIM PARAMETRELERI
# ----------------------------------------------------------------
# MobileNetV2'nin ImageNet on-egitim cozunurlugu 224x224'tur.
GORUNTU_BOYUTU = (224, 224)
# Renk kanali sayisi (RGB).
KANAL_SAYISI = 3
# Model giris sekli (yukseklik, genislik, kanal).
GIRIS_SEKLI = (GORUNTU_BOYUTU[0], GORUNTU_BOYUTU[1], KANAL_SAYISI)

# Bellek/hiz dengesi icin makul bir yigin boyutu.
YIGIN_BOYUTU = 32

# Egitim icinden ayrilacak dogrulama (validation) orani.
DOGRULAMA_ORANI = 0.2

# Tekrarlanabilirlik icin sabit rastgelelik tohumu.
TOHUM = 42

# ----------------------------------------------------------------
# IKI ASAMALI EGITIM AYARLARI
# ----------------------------------------------------------------
# Asama 1: Ozellik cikarimi (base model donuk).
ASAMA1_EPOCH = 20
ASAMA1_OGRENME_ORANI = 1e-3

# Asama 2: Ince ayar (fine-tuning, ust katmanlar acik).
ASAMA2_EPOCH = 10
ASAMA2_OGRENME_ORANI = 1e-5
# Base modelin sondan kac katmaninin egitilebilir yapilacagi.
INCE_AYAR_KATMAN_SAYISI = 30

# ----------------------------------------------------------------
# GERI CAGRIM (CALLBACK) AYARLARI
# ----------------------------------------------------------------
# EarlyStopping: dogrulama kaybi iyilesmezse egitimi durdurur.
ERKEN_DURDURMA_SABIR = 5
ERKEN_DURDURMA_IZLE = "val_loss"
# ModelCheckpoint: en iyi dogrulama dogrulugunu kaydeder.
CHECKPOINT_IZLE = "val_accuracy"

# ----------------------------------------------------------------
# GRAD-CAM
# ----------------------------------------------------------------
# Isi haritasinin orijinal goruntu uzerine bindirilme siddeti (0-1).
GRADCAM_ALFA = 0.4

# ----------------------------------------------------------------
# OOD (OUT-OF-DISTRIBUTION) AYARLARI
# ----------------------------------------------------------------
# Her sinif icin embedding istatistigi cikarilirken kullanilacak azami
# ornek sayisi. Daha yuksek deger daha stabil esik, daha uzun sure demektir.
OOD_SINIF_BASI_AZAMI_ORNEK = 400

# Mesafe-oncelikli UC KADEMELI OOD karari:
#   mesafe > OOD_MESAFE_T_YUKSEK -> kosulsuz "Bilinmeyen" (softmax guveni ezemez)
#   mesafe < OOD_MESAFE_T_DUSUK  -> "Bilinen"
#   gri bolge (arasi)            -> softmax sinifi ile en yakin merkez sinifi
#                                   AYNIYSA ve guven kosullari saglaniyorsa
#                                   "Sinirda" (tahmin dusuk guven uyarisiyla
#                                   gosterilir); aksi halde "Bilinmeyen".
# Gerekce: gercek ama veri setine benzemeyen meyveler (orn. sari burusuk elma,
# kNN mesafesi 0.41) ile yabanci meyveler (murdum 0.45, uzum 0.50) embedding
# uzayinda ic ice gectiginden tek esikle ayrilamaz. Uc kademeli karar, sinirda
# kalan vakalarda tahmini gizlemek yerine acik bir belirsizlik uyarisi verir.
# Train dagilimi P95~0.41; T_dusuk bunun ustu veri seti orneklerini "bilinen"
# yapar. T_yuksek dis OOD'yi yakalar; aradaki dar gri bolge sinirda uyarisi.
OOD_MESAFE_T_YUKSEK = 0.485
OOD_MESAFE_T_DUSUK = 0.365
# Gri bolgede softmax-merkez uyumu + guven kosulu.
OOD_GRI_MAXPROB_ESIK = 0.80
OOD_GRI_ENTROPY_ESIK = 0.30
# Gri bolgede merkez uyumsuz olsa bile (orn. sari elma) yuksek softmax
# guvenine izin veren esik; dusuk guvenli yabanci meyveler bilinmeyen kalir.
OOD_GRI_YUKSEK_GUVEN = 0.88

# Temperature scaling (opsiyonel guven kalibrasyonu) parametre dosyasi.
# Tahmin sinifini DEGISTIRMEZ; yalnizca guven yuzdesini gercekci olceklendirir.
KALIBRASYON_DOSYA = MODEL_DIZIN / "temperature_calibration.json"


def klasorleri_hazirla():
    """Cikti klasorlerinin var oldugundan emin olur, yoksa olusturur."""
    for klasor in (MODEL_DIZIN, CIKTI_DIZIN, RAPOR_DIZIN, GRADCAM_DIZIN):
        klasor.mkdir(parents=True, exist_ok=True)
