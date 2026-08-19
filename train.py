"""Egitim akisi (Iki Asamali Transfer Learning).

Bu betik:
1. Veriyi yukler ve sinif isimlerini dinamik belirler.
2. MobileNetV2 tabanli modeli kurar.
3. Asama 1'de yeni siniflandirma basligini egitir (taban donuk).
4. Asama 2'de tabanin ust katmanlarini acip ince ayar yapar.
5. En iyi modeli, sinif indekslerini ve grafikleri kaydeder.

Kullanim:
    python train.py
"""

import tensorflow as tf

from src import config
from src.data_loader import (
    egitim_dogrulama_yukle,
    sinif_agirliklari_hesapla,
    sinif_indekslerini_kaydet,
)
from src.model_builder import ince_ayar_ic_acan, model_olustur
from src.utils import egitim_gecmisini_kaydet, egrileri_ciz, gecmisleri_birlestir


def geri_cagrimlar_olustur():
    """EarlyStopping ve ModelCheckpoint geri cagrimlarini hazirlar."""
    erken_durdurma = tf.keras.callbacks.EarlyStopping(
        monitor=config.ERKEN_DURDURMA_IZLE,
        patience=config.ERKEN_DURDURMA_SABIR,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.MODEL_DOSYA),
        monitor=config.CHECKPOINT_IZLE,
        save_best_only=True,
        verbose=1,
    )
    return [erken_durdurma, checkpoint]


def egit():
    """Tum egitim akisini calistirir."""
    # Tekrarlanabilirlik icin tohumu sabitle.
    tf.keras.utils.set_random_seed(config.TOHUM)
    config.klasorleri_hazirla()

    print("[1/6] Veri yukleniyor...")
    egitim_ds, dogrulama_ds, sinif_isimleri = egitim_dogrulama_yukle()
    sinif_sayisi = len(sinif_isimleri)
    print(f"  Tespit edilen siniflar ({sinif_sayisi}): {sinif_isimleri}")

    # Etiket kaymasini onlemek icin sinif indekslerini hemen kaydet.
    sinif_indekslerini_kaydet(sinif_isimleri)

    print("[2/6] Sinif agirliklari hesaplaniyor (dengesizlik telafisi)...")
    sinif_agirliklari = sinif_agirliklari_hesapla()
    print(f"  Sinif agirliklari: {sinif_agirliklari}")

    print("[3/6] Model kuruluyor (MobileNetV2 + transfer learning)...")
    model, base_model = model_olustur(sinif_sayisi, artirma_kullan=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.ASAMA1_OGRENME_ORANI),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    geri_cagrimlar = geri_cagrimlar_olustur()

    print("[4/6] ASAMA 1 - Ozellik cikarimi egitimi basliyor...")
    gecmis1 = model.fit(
        egitim_ds,
        validation_data=dogrulama_ds,
        epochs=config.ASAMA1_EPOCH,
        class_weight=sinif_agirliklari,
        callbacks=geri_cagrimlar,
        verbose=1,
    )

    print("[5/6] ASAMA 2 - Ince ayar (fine-tuning) basliyor...")
    ince_ayar_ic_acan(base_model)
    # Ince ayarda cok dusuk ogrenme orani kullanilir (onceden ogrenileni
    # bozmamak icin).
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.ASAMA2_OGRENME_ORANI),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    gecmis2 = model.fit(
        egitim_ds,
        validation_data=dogrulama_ds,
        epochs=config.ASAMA2_EPOCH,
        class_weight=sinif_agirliklari,
        callbacks=geri_cagrimlar,
        verbose=1,
    )

    print("[6/6] Sonuclar kaydediliyor...")
    # NOT:
    # ModelCheckpoint zaten en iyi val_accuracy modelini kaydetmektedir.
    # Burada tekrar model.save() cagrisi yapilirsa en iyi modelin
    # ustune son epoch agirliklari yazilabilir. Bu nedenle ek kayit yok.

    birlesik_gecmis = gecmisleri_birlestir(gecmis1, gecmis2)
    egitim_gecmisini_kaydet(birlesik_gecmis)
    egrileri_ciz(birlesik_gecmis, asama1_epoch=len(gecmis1.history["loss"]))

    print("\nEgitim tamamlandi.")
    print(f"  Model      : {config.MODEL_DOSYA}")
    print(f"  Siniflar   : {config.SINIF_INDEKS_DOSYA}")
    print(f"  Grafikler  : {config.CIKTI_DIZIN}")


if __name__ == "__main__":
    egit()
