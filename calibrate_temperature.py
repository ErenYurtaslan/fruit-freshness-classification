"""Temperature scaling (guven kalibrasyonu) betigi.

Temperature scaling, modelin AGIRLIKLARINI ve TAHMIN SINIFINI degistirmez;
yalnizca softmax olasiliklarini tek bir T parametresiyle yumusatarak guven
yuzdesini gercekci hale getirir:

    kalibre_olasilik = softmax(log(p) / T)

log(softmax(z)) = z - logsumexp(z) oldugu icin log-olasiliklar uzerinden
bolme, logitler uzerinden bolmeyle ayni siralamayi ve ayni argmax'i verir.
T > 1 ise asiri ozguven azalir.

T, dogrulama (validation) kumesi uzerinde negatif log-likelihood (NLL)
minimize edilerek ogrenilir (tek skaler, model egitimi yok). Sonuc
models/temperature_calibration.json dosyasina kaydedilir ve uygulama
arayuzunde "kalibre guven" olarak gosterilir.

Kullanim:
    python calibrate_temperature.py
"""

import json
from datetime import datetime

import numpy as np
import tensorflow as tf

from src import config
from src.data_loader import egitim_dogrulama_yukle


def kalibre_olasiliklar(olasiliklar: np.ndarray, t: float) -> np.ndarray:
    """Softmax olasiliklarini T sicakligiyla yeniden olcekler (argmax sabit)."""
    eps = 1e-12
    log_p = np.log(np.clip(olasiliklar, eps, 1.0)) / t
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    return p / p.sum(axis=1, keepdims=True)


def nll_hesapla(olasiliklar: np.ndarray, etiketler: np.ndarray) -> float:
    """Ortalama negatif log-likelihood."""
    eps = 1e-12
    dogru_p = olasiliklar[np.arange(len(etiketler)), etiketler]
    return float(-np.mean(np.log(np.clip(dogru_p, eps, 1.0))))


def ece_hesapla(olasiliklar: np.ndarray, etiketler: np.ndarray, kutu_sayisi: int = 15) -> float:
    """Expected Calibration Error: guven ile gercek dogruluk arasindaki fark."""
    guven = olasiliklar.max(axis=1)
    tahmin = olasiliklar.argmax(axis=1)
    dogru = (tahmin == etiketler).astype(float)

    ece = 0.0
    sinirlar = np.linspace(0.0, 1.0, kutu_sayisi + 1)
    for alt, ust in zip(sinirlar[:-1], sinirlar[1:]):
        maske = (guven > alt) & (guven <= ust)
        if maske.sum() == 0:
            continue
        ece += (maske.mean()) * abs(dogru[maske].mean() - guven[maske].mean())
    return float(ece)


def main():
    config.klasorleri_hazirla()

    print("Model yukleniyor...")
    model = tf.keras.models.load_model(str(config.MODEL_DOSYA))

    print("Dogrulama kumesi hazirlaniyor (egitimle ayni tohum/ayrim)...")
    _, dogrulama_ds, _ = egitim_dogrulama_yukle()

    print("Dogrulama tahminleri toplaniyor...")
    olasiliklar, etiketler = [], []
    for x, y in dogrulama_ds:
        olasiliklar.append(model.predict(x, verbose=0))
        etiketler.append(y.numpy())
    olasiliklar = np.concatenate(olasiliklar, axis=0)
    etiketler = np.concatenate(etiketler, axis=0)
    print(f"Toplam dogrulama ornegi: {len(etiketler)}")

    # T icin izgara aramasi: tek skaler parametre, kapali form yok.
    adaylar = np.arange(0.50, 5.01, 0.05)
    nll_degerleri = [
        nll_hesapla(kalibre_olasiliklar(olasiliklar, float(t)), etiketler)
        for t in adaylar
    ]
    en_iyi_t = float(adaylar[int(np.argmin(nll_degerleri))])

    nll_once = nll_hesapla(olasiliklar, etiketler)
    nll_sonra = float(np.min(nll_degerleri))
    ece_once = ece_hesapla(olasiliklar, etiketler)
    ece_sonra = ece_hesapla(kalibre_olasiliklar(olasiliklar, en_iyi_t), etiketler)

    # Argmax degismedigini dogrula (temperature scaling garantisi).
    ayni_argmax = bool(
        np.array_equal(
            olasiliklar.argmax(axis=1),
            kalibre_olasiliklar(olasiliklar, en_iyi_t).argmax(axis=1),
        )
    )

    sonuc = {
        "T": en_iyi_t,
        "olusturma_zamani": datetime.now().isoformat(timespec="seconds"),
        "dogrulama_ornek_sayisi": int(len(etiketler)),
        "nll_once": nll_once,
        "nll_sonra": nll_sonra,
        "ece_once": ece_once,
        "ece_sonra": ece_sonra,
        "argmax_degisti_mi": not ayni_argmax,
    }
    config.KALIBRASYON_DOSYA.write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nEn iyi T: {en_iyi_t:.2f}")
    print(f"NLL: {nll_once:.4f} -> {nll_sonra:.4f}")
    print(f"ECE: {ece_once:.4f} -> {ece_sonra:.4f}")
    print(f"Argmax degisti mi: {not ayni_argmax}")
    print(f"Kayit: {config.KALIBRASYON_DOSYA}")


if __name__ == "__main__":
    main()
