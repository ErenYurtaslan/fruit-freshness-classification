"""Grad-CAM (Gradient-weighted Class Activation Mapping) modulu.

Grad-CAM, modelin bir karari verirken goruntunun HANGI bolgelerine
odaklandigini isi haritasi olarak gosterir. Boylece "kara kutu" model
aciklanabilir hale gelir (Aciklanabilir Yapay Zeka - XAI).

Akis:
1. Modelin son konvolusyon katmani OTOMATIK tespit edilir (gomulmez).
2. Hedef sinif skorunun bu katman ciktilarina gore gradyani alinir.
3. Gradyanlar kanal bazinda ortalanip agirlik olarak kullanilir.
4. Agirlikli kanal toplami + ReLU ile isi haritasi olusur.
5. Harita orijinal goruntuye bindirilir (overlay).

Kullanim:
    python gradcam.py --image dataset/test/rottenapples/ornek.png
"""

import argparse

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from src import config  # noqa: E402
from src.data_loader import sinif_indekslerini_oku  # noqa: E402
from src.utils import goruntu_yukle_ve_hazirla  # noqa: E402


def _ic_model_ve_son_conv_bul(model):
    """Modeldeki ic (taban) modeli ve son konvolusyon katmanini bulur.

    MobileNetV2 transfer learning modelinde konvolusyon katmanlari ic
    icine yerlestirilmis taban model icinde bulunur. Dikkat: veri artirma
    katmani da bir Sequential (yani Keras Model) oldugundan, taban model
    olarak ICINDE Conv2D barindiran model secilir. Son konvolusyon katmani
    da Conv2D tipine gore (sondan basa) otomatik tespit edilir.

    Donen deger: (taban_model, son_conv_katman)
    """
    from tensorflow.keras.layers import Conv2D

    taban_model = None
    for katman in model.layers:
        if isinstance(katman, tf.keras.Model):
            # Yalnizca konvolusyon iceren ic modeli sec (augmentation'i atla).
            if any(isinstance(alt, Conv2D) for alt in katman.layers):
                taban_model = katman
                break

    # Ic model yoksa son conv'u dogrudan ana modelde ara.
    aranacak = taban_model if taban_model is not None else model

    son_conv = None
    for katman in reversed(aranacak.layers):
        if isinstance(katman, Conv2D):
            son_conv = katman
            break

    if son_conv is None:
        raise ValueError("Son konvolusyon katmani tespit edilemedi.")

    return aranacak, son_conv


def grad_model_olustur(model):
    """Grad-CAM icin gradyan modelini olusturur.

    Cikislar: [son konvolusyon aktivasyonu, model tahminleri].
    Her ikisi de ayni hesaplama grafiginde uretildiginden, tahmin
    skorunun konvolusyon aktivasyonuna gore gradyani alinabilir.
    """
    taban_model, son_conv = _ic_model_ve_son_conv_bul(model)

    # Tahminleri tabanin ciktisindan, ana modelin ust katmanlarini
    # yeniden kullanarak olustur.
    taban_cikti = taban_model.output
    x = taban_cikti
    # Ic modelden sonra gelen ust katmanlari sirayla uygula.
    ic_model_gecildi = False
    for katman in model.layers:
        if isinstance(katman, tf.keras.Model):
            ic_model_gecildi = True
            continue
        if ic_model_gecildi:
            x = katman(x)

    grad_model = tf.keras.models.Model(
        inputs=taban_model.inputs,
        outputs=[son_conv.output, x],
    )
    return grad_model, son_conv.name


def isi_haritasi_uret(grad_model, model_girisi, sinif_indeksi=None):
    """Verilen goruntu icin Grad-CAM isi haritasi (0-1) uretir.

    Parametreler:
        grad_model: grad_model_olustur ile uretilen model.
        model_girisi: on islenmis, yigin boyutu eklenmis goruntu.
        sinif_indeksi: None ise en yuksek olasilikli sinif kullanilir.

    Donen deger: (isi_haritasi, tahmin_indeksi, olasiliklar)
    """
    with tf.GradientTape() as tape:
        conv_aktivasyon, tahminler = grad_model(model_girisi)
        if sinif_indeksi is None:
            sinif_indeksi = int(tf.argmax(tahminler[0]))
        hedef_skor = tahminler[:, sinif_indeksi]

    # Hedef skorun konvolusyon aktivasyonlarina gore gradyani.
    gradyanlar = tape.gradient(hedef_skor, conv_aktivasyon)

    # Her kanalin onemini gradyanlarin uzaysal ortalamasi belirler.
    havuzlanmis_gradyanlar = tf.reduce_mean(gradyanlar, axis=(0, 1, 2))

    conv_aktivasyon = conv_aktivasyon[0]
    isi_haritasi = conv_aktivasyon @ havuzlanmis_gradyanlar[..., tf.newaxis]
    isi_haritasi = tf.squeeze(isi_haritasi)

    # Yalnizca pozitif katkilar (ReLU) ve 0-1 normalize.
    isi_haritasi = tf.maximum(isi_haritasi, 0)
    en_buyuk = tf.reduce_max(isi_haritasi)
    if en_buyuk > 0:
        isi_haritasi = isi_haritasi / en_buyuk

    return isi_haritasi.numpy(), int(sinif_indeksi), tahminler.numpy()[0]


def overlay_olustur(ham_goruntu, isi_haritasi, alfa=None):
    """Isi haritasini orijinal goruntu uzerine bindirir (overlay).

    Parametreler:
        ham_goruntu: 0-255 araliginda RGB uint8 dizi.
        isi_haritasi: 0-1 araliginda isi haritasi.
        alfa: bindirme siddeti.

    Donen deger: uint8 RGB overlay goruntusu.
    """
    alfa = alfa if alfa is not None else config.GRADCAM_ALFA
    yukseklik, genislik = ham_goruntu.shape[:2]

    # Isi haritasini goruntu boyutuna olcekle ve renklendir.
    isi = cv2.resize(isi_haritasi, (genislik, yukseklik))
    isi = np.uint8(255 * isi)
    isi_renkli = cv2.applyColorMap(isi, cv2.COLORMAP_JET)
    # OpenCV BGR uretir; RGB'ye cevir.
    isi_renkli = cv2.cvtColor(isi_renkli, cv2.COLOR_BGR2RGB)

    overlay = isi_renkli * alfa + ham_goruntu * (1 - alfa)
    return np.uint8(overlay)


def gradcam_uret_ve_kaydet(goruntu_yolu, model=None, kayit_yolu=None):
    """Bir goruntu icin Grad-CAM uretip diske kaydeder.

    model verilmezse diskten yuklenir (CLI kullanimi). app.py gibi
    cagiranlar performans icin onceden yuklenmis modeli gecebilir.

    Donen deger: (kayit_yolu, tahmin_sinifi, guven_skoru)
    """
    config.klasorleri_hazirla()

    if model is None:
        if not config.MODEL_DOSYA.exists():
            raise FileNotFoundError(
                "Egitilmis model bulunamadi. Once 'python train.py' calistirin."
            )
        model = tf.keras.models.load_model(str(config.MODEL_DOSYA))

    sinif_isimleri = sinif_indekslerini_oku()
    model_girisi, ham_goruntu = goruntu_yukle_ve_hazirla(goruntu_yolu)

    grad_model, conv_ad = grad_model_olustur(model)
    isi_haritasi, sinif_indeksi, olasiliklar = isi_haritasi_uret(
        grad_model, model_girisi
    )
    overlay = overlay_olustur(ham_goruntu, isi_haritasi)

    tahmin_sinifi = sinif_isimleri[sinif_indeksi]
    guven = float(olasiliklar[sinif_indeksi])

    # Karsilastirmali gorseli kaydet (orijinal + isi haritasi + overlay).
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(ham_goruntu)
    plt.title("Orijinal Goruntu")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(isi_haritasi, cmap="jet")
    plt.title(f"Isi Haritasi\n(katman: {conv_ad})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Tahmin: {tahmin_sinifi}\nGuven: {guven * 100:.1f}%")
    plt.axis("off")

    plt.tight_layout()

    if kayit_yolu is None:
        from pathlib import Path

        ad = Path(goruntu_yolu).stem
        kayit_yolu = config.GRADCAM_DIZIN / f"gradcam_{ad}.png"
    plt.savefig(kayit_yolu, dpi=150, bbox_inches="tight")
    plt.close()

    return str(kayit_yolu), tahmin_sinifi, guven


def _argumanlari_oku():
    """Komut satiri argumanlarini okur."""
    ayristirici = argparse.ArgumentParser(
        description="Grad-CAM isi haritasi uretir."
    )
    ayristirici.add_argument(
        "--image", required=True, help="Grad-CAM uretilecek goruntu yolu."
    )
    return ayristirici.parse_args()


if __name__ == "__main__":
    argumanlar = _argumanlari_oku()
    yol, sinif, guven = gradcam_uret_ve_kaydet(argumanlar.image)
    print(f"Grad-CAM kaydedildi: {yol}")
    print(f"Tahmin: {sinif} (guven: {guven * 100:.1f}%)")
