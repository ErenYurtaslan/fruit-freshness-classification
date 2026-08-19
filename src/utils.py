"""Yardimci fonksiyonlar modulu.

Grafik cizimi, gecmis birlestirme ve goruntu yukleme gibi birden cok
dosyada tekrar eden islemleri merkezilestirir.
"""

import json

import matplotlib

# Arayuz olmayan ortamlarda (sunucu/CLI) grafik kaydedebilmek icin.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from . import config  # noqa: E402


def gecmisleri_birlestir(gecmis1, gecmis2=None):
    """Iki egitim asamasinin Keras History sozluklerini birlestirir.

    Asama 1 ve Asama 2 (ince ayar) metriklerini tek bir surekli egri
    olarak gostermek icin kullanilir.
    """
    birlesik = {}
    anahtarlar = gecmis1.history.keys()
    for anahtar in anahtarlar:
        degerler = list(gecmis1.history[anahtar])
        if gecmis2 is not None and anahtar in gecmis2.history:
            degerler += list(gecmis2.history[anahtar])
        birlesik[anahtar] = degerler
    return birlesik


def egitim_gecmisini_kaydet(gecmis_sozluk):
    """Birlesik egitim gecmisini JSON olarak kaydeder."""
    config.klasorleri_hazirla()
    with open(config.EGITIM_GECMIS_DOSYA, "w", encoding="utf-8") as dosya:
        json.dump(gecmis_sozluk, dosya, ensure_ascii=False, indent=2)


def egrileri_ciz(gecmis_sozluk, asama1_epoch=None):
    """Dogruluk (accuracy) ve kayip (loss) egrilerini cizip kaydeder.

    Iki ayri PNG uretir: outputs/accuracy.png ve outputs/loss.png.
    Ince ayarin basladigi epoch dikey kesik cizgi ile isaretlenir.
    """
    config.klasorleri_hazirla()

    epoch_sayisi = len(gecmis_sozluk.get("loss", []))
    epoch_ekseni = range(1, epoch_sayisi + 1)

    # --- Dogruluk grafigi ---
    plt.figure(figsize=(8, 5))
    if "accuracy" in gecmis_sozluk:
        plt.plot(epoch_ekseni, gecmis_sozluk["accuracy"], label="Egitim Dogrulugu")
    if "val_accuracy" in gecmis_sozluk:
        plt.plot(
            epoch_ekseni, gecmis_sozluk["val_accuracy"], label="Dogrulama Dogrulugu"
        )
    if asama1_epoch:
        plt.axvline(
            x=asama1_epoch + 0.5,
            color="gray",
            linestyle="--",
            label="Ince Ayar Baslangici",
        )
    plt.title("Model Dogrulugu (Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Dogruluk")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.CIKTI_DIZIN / "accuracy.png", dpi=150)
    plt.close()

    # --- Kayip grafigi ---
    plt.figure(figsize=(8, 5))
    if "loss" in gecmis_sozluk:
        plt.plot(epoch_ekseni, gecmis_sozluk["loss"], label="Egitim Kaybi")
    if "val_loss" in gecmis_sozluk:
        plt.plot(epoch_ekseni, gecmis_sozluk["val_loss"], label="Dogrulama Kaybi")
    if asama1_epoch:
        plt.axvline(
            x=asama1_epoch + 0.5,
            color="gray",
            linestyle="--",
            label="Ince Ayar Baslangici",
        )
    plt.title("Model Kaybi (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Kayip")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.CIKTI_DIZIN / "loss.png", dpi=150)
    plt.close()


def goruntu_yukle_ve_hazirla(goruntu_yolu):
    """Tek bir goruntuyu modele uygun bicimde yukler.

    Donen deger:
        (model_girisi, ham_dizi)
        - model_girisi: on islenmis, yigin boyutu eklenmis tensor.
        - ham_dizi: 0-255 araligindaki orijinal RGB dizi (gorsellestirme icin).
    """
    goruntu = tf.keras.utils.load_img(
        goruntu_yolu, target_size=config.GORUNTU_BOYUTU
    )
    ham_dizi = tf.keras.utils.img_to_array(goruntu)
    model_girisi = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.copy(ham_dizi)
    )
    model_girisi = np.expand_dims(model_girisi, axis=0)
    return model_girisi, ham_dizi.astype("uint8")
