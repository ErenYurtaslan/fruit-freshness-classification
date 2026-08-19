"""Model kurulum modulu (MobileNetV2 + Transfer Learning).

Sorumlulugu: Onceden ImageNet uzerinde egitilmis MobileNetV2 tabanini
kullanarak meyve tazelik siniflandirmasi icin yeni bir model olusturur.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from . import config
from .data_loader import artirma_katmani_olustur


def model_olustur(sinif_sayisi, artirma_kullan=True):
    """MobileNetV2 tabanli transfer learning modeli olusturur.

    Mimari akis:
        Giris -> (veri artirma) -> MobileNetV2 tabani (donuk)
              -> GlobalAveragePooling -> Dropout -> Dense(softmax)

    Parametreler:
        sinif_sayisi: Cikis katmanindaki noron (sinif) sayisi.
        artirma_kullan: Egitimde veri artirma katmaninin eklenip
            eklenmeyecegi. Tahmin/degerlendirmede False onerilir.

    Donen deger: (model, base_model)
        base_model ileride ince ayar (fine-tuning) icin gereklidir.
    """
    girisler = layers.Input(shape=config.GIRIS_SEKLI, name="giris")

    x = girisler
    if artirma_kullan:
        x = artirma_katmani_olustur()(x)

    # ImageNet agirliklariyla MobileNetV2 tabani; ust siniflandirma yok.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=config.GIRIS_SEKLI,
        include_top=False,
        weights="imagenet",
    )
    # Asama 1'de taban donuk tutulur (yalnizca yeni baslik egitilir).
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_havuz")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    ciktilar = layers.Dense(
        sinif_sayisi, activation="softmax", name="siniflandirma"
    )(x)

    model = models.Model(girisler, ciktilar, name="meyve_tazelik_modeli")
    return model, base_model


def ince_ayar_ic_acan(base_model, katman_sayisi=None):
    """Ince ayar (fine-tuning) icin base modelin ust katmanlarini acar.

    Tabanin tamami yerine yalnizca sondan 'katman_sayisi' kadar katman
    egitilebilir yapilir; boylece ImageNet'ten gelen dusuk seviyeli
    ozellikler korunur, yalnizca yuksek seviyeli ozellikler uyarlanir.
    """
    katman_sayisi = katman_sayisi or config.INCE_AYAR_KATMAN_SAYISI
    base_model.trainable = True
    # Belirtilen sayidaki son katman haricindekileri tekrar dondur.
    for katman in base_model.layers[:-katman_sayisi]:
        katman.trainable = False
