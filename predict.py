"""Tahmin akisi (Tek goruntu uzerinde siniflandirma).

Hem komut satirindan (CLI) hem de PyQt5 arayuzunden (app.py) ortak
kullanilabilecek tahmin fonksiyonlari saglar.

Kullanim:
    python predict.py --image dataset/test/freshapples/ornek.png
"""

import argparse

import numpy as np
import tensorflow as tf

from src import config
from src.data_loader import sinif_indekslerini_oku
from src.utils import goruntu_yukle_ve_hazirla


def model_yukle():
    """Egitilmis modeli diskten yukler.

    app.py bu fonksiyonu uygulama acilisinda BIR KEZ cagirarak modeli
    bellekte tutar; boylece her tahminde yeniden yukleme (ve arayuz
    donmasi) onlenir.
    """
    if not config.MODEL_DOSYA.exists():
        raise FileNotFoundError(
            "Egitilmis model bulunamadi. Once 'python train.py' calistirin."
        )
    return tf.keras.models.load_model(str(config.MODEL_DOSYA))


def tahmin_et(goruntu_yolu, model=None):
    """Tek bir goruntu icin tahmin yapar.

    Donen deger sozlugu:
        {
            "sinif": en yuksek olasilikli sinif adi,
            "guven": o sinifin olasiligi (0-1),
            "olasiliklar": {sinif_adi: olasilik, ...}
        }
    """
    if model is None:
        model = model_yukle()

    sinif_isimleri = sinif_indekslerini_oku()
    model_girisi, _ = goruntu_yukle_ve_hazirla(goruntu_yolu)

    olasiliklar = model.predict(model_girisi, verbose=0)[0]
    en_iyi_indeks = int(np.argmax(olasiliklar))

    return {
        "sinif": sinif_isimleri[en_iyi_indeks],
        "guven": float(olasiliklar[en_iyi_indeks]),
        "olasiliklar": {
            ad: float(olasilik)
            for ad, olasilik in zip(sinif_isimleri, olasiliklar)
        },
    }


def _argumanlari_oku():
    """Komut satiri argumanlarini okur."""
    ayristirici = argparse.ArgumentParser(
        description="Tek goruntu uzerinde meyve tazelik tahmini yapar."
    )
    ayristirici.add_argument(
        "--image", required=True, help="Tahmin edilecek goruntunun yolu."
    )
    return ayristirici.parse_args()


def _sonucu_yazdir(sonuc):
    """Tahmin sonucunu okunabilir bicimde konsola yazar."""
    print("=" * 45)
    print(f"  TAHMIN     : {sonuc['sinif']}")
    print(f"  GUVEN SKORU: {sonuc['guven'] * 100:.2f}%")
    print("-" * 45)
    print("  Tum Sinif Olasiliklari:")
    # Olasiliga gore azalan sirada listele.
    for ad, olasilik in sorted(
        sonuc["olasiliklar"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"    {ad:<16}: {olasilik * 100:6.2f}%")
    print("=" * 45)


if __name__ == "__main__":
    argumanlar = _argumanlari_oku()
    sonuc = tahmin_et(argumanlar.image)
    _sonucu_yazdir(sonuc)
