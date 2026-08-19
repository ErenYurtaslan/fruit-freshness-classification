"""Veri yukleme ve hazirlama modulu.

Sorumluluklari:
- Sinif isimlerini dataset klasorunden DINAMIK olarak okur (koda gomulmez).
- Egitim/dogrulama/test veri kumelerini (tf.data.Dataset) hazirlar.
- Hafif veri artirma (augmentation) katmani saglar.
- Sinif indekslerini diske kaydeder/okur (etiket kaymasini onler).
- Sinif dengesizligi icin sinif agirliklarini hesaplar.
"""

import json
from collections import Counter

import tensorflow as tf
from tensorflow.keras import layers

from . import config


def sinif_isimlerini_belirle(dizin=None):
    """Verilen dizindeki alt klasor adlarini alfabetik sirada dondurur.

    Klasorden okundugu icin sinif isimleri sabit degildir; veri setinin
    gercek yapisina gore otomatik belirlenir.
    """
    dizin = dizin or config.EGITIM_DIZIN
    sinif_isimleri = sorted(
        [p.name for p in dizin.iterdir() if p.is_dir()]
    )
    if not sinif_isimleri:
        raise FileNotFoundError(
            f"'{dizin}' icinde sinif klasoru bulunamadi."
        )
    return sinif_isimleri


def _on_isleme(goruntu, etiket):
    """MobileNetV2'nin bekledigi on islemeyi uygular ([-1, 1] olceklemesi)."""
    goruntu = tf.cast(goruntu, tf.float32)
    goruntu = tf.keras.applications.mobilenet_v2.preprocess_input(goruntu)
    return goruntu, etiket


def artirma_katmani_olustur():
    """Hafif veri artirma katmani olusturur.

    Veri seti ZATEN artirilmis oldugundan agresif augmentation
    bilgi bozulmasina yol acabilir; bu yuzden hafif tutulur.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=config.TOHUM),
            layers.RandomRotation(0.1, seed=config.TOHUM),
            layers.RandomZoom(0.1, seed=config.TOHUM),
            layers.RandomContrast(0.1, seed=config.TOHUM),
        ],
        name="veri_artirma",
    )


def egitim_dogrulama_yukle():
    """Egitim ve dogrulama veri kumelerini hazirlar.

    Hazir bir validation klasoru olmadigi icin egitim verisinden
    config.DOGRULAMA_ORANI kadari ayrilir. Ayni tohum kullanilarak
    egitim/dogrulama ayrimi tekrarlanabilir tutulur.

    Donen deger: (egitim_ds, dogrulama_ds, sinif_isimleri)
    """
    ortak = dict(
        directory=config.EGITIM_DIZIN,
        validation_split=config.DOGRULAMA_ORANI,
        seed=config.TOHUM,
        image_size=config.GORUNTU_BOYUTU,
        batch_size=config.YIGIN_BOYUTU,
        label_mode="int",
    )

    egitim_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training", shuffle=True, **ortak
    )
    dogrulama_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation", shuffle=False, **ortak
    )

    # image_dataset_from_directory sinif isimlerini alfabetik verir.
    sinif_isimleri = egitim_ds.class_names

    # On isleme ve performans optimizasyonlarini uygula.
    otomatik = tf.data.AUTOTUNE
    egitim_ds = (
        egitim_ds.map(_on_isleme, num_parallel_calls=otomatik)
        .prefetch(otomatik)
    )
    dogrulama_ds = (
        dogrulama_ds.map(_on_isleme, num_parallel_calls=otomatik)
        .prefetch(otomatik)
    )

    return egitim_ds, dogrulama_ds, sinif_isimleri


def test_yukle():
    """Test veri kumesini hazirlar (yalnizca final degerlendirme icin).

    Donen deger: (test_ds, sinif_isimleri)
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=config.TEST_DIZIN,
        image_size=config.GORUNTU_BOYUTU,
        batch_size=config.YIGIN_BOYUTU,
        label_mode="int",
        shuffle=False,
    )
    sinif_isimleri = test_ds.class_names
    otomatik = tf.data.AUTOTUNE
    test_ds = test_ds.map(_on_isleme, num_parallel_calls=otomatik).prefetch(
        otomatik
    )
    return test_ds, sinif_isimleri


def sinif_agirliklari_hesapla():
    """Egitim klasorundeki goruntu sayilarina gore sinif agirliklarini hesaplar.

    Az ornekli siniflarin (orn. freshoranges) ogrenilmesini guclendirmek
    icin dengesiz veriyi telafi eder. sklearn'in 'balanced' formuluyle ayni:
        agirlik = toplam / (sinif_sayisi * o_sinifin_ornek_sayisi)

    Donen deger: {sinif_indeksi: agirlik} sozlugu.
    """
    sinif_isimleri = sinif_isimlerini_belirle()
    sayimlar = Counter()
    for indeks, ad in enumerate(sinif_isimleri):
        klasor = config.EGITIM_DIZIN / ad
        adet = sum(1 for _ in klasor.iterdir() if _.is_file())
        sayimlar[indeks] = adet

    toplam = sum(sayimlar.values())
    sinif_sayisi = len(sinif_isimleri)
    agirliklar = {
        indeks: toplam / (sinif_sayisi * adet)
        for indeks, adet in sayimlar.items()
    }
    return agirliklar


def sinif_indekslerini_kaydet(sinif_isimleri):
    """Sinif indeks -> isim eslemesini JSON olarak kaydeder.

    Tahmin ve degerlendirme asamalarinda etiket sirasinin kaymamasi
    icin kritik onemdedir.
    """
    config.klasorleri_hazirla()
    esleme = {indeks: ad for indeks, ad in enumerate(sinif_isimleri)}
    with open(config.SINIF_INDEKS_DOSYA, "w", encoding="utf-8") as dosya:
        json.dump(esleme, dosya, ensure_ascii=False, indent=2)


def sinif_indekslerini_oku():
    """Kaydedilmis sinif indeks -> isim eslemesini okur.

    Donen deger: indeks sirasina gore sinif isimleri listesi.
    """
    if not config.SINIF_INDEKS_DOSYA.exists():
        raise FileNotFoundError(
            "class_indices.json bulunamadi. Once train.py calistirilmali."
        )
    with open(config.SINIF_INDEKS_DOSYA, "r", encoding="utf-8") as dosya:
        esleme = json.load(dosya)
    # Anahtarlar string olarak gelir; integer'a cevirip siralayalim.
    return [esleme[str(i)] for i in range(len(esleme))]
