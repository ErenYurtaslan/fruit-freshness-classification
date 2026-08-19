"""Degerlendirme akisi (Test seti uzerinde performans olcumu).

Bu betik egitilmis modeli test verisi uzerinde calistirir ve su
ciktilari uretir:
- Accuracy, Precision, Recall, F1 (makro ve agirlikli)
- Classification Report (sinif bazinda) -> reports/classification_report.txt
- Confusion Matrix gorseli -> outputs/confusion_matrix.png
- Tum metrikleri ozetleyen JSON -> reports/metrikler.json

Metriklerin rolu:
- Accuracy : Genel dogruluk ozeti.
- Precision: Yanlis pozitif maliyeti (curugu taze sanma riski).
- Recall   : Kacirilan gercek vakalar (curugu yakalayamama).
- F1       : Dengesiz veride precision/recall dengesi.

Kullanim:
    python evaluate.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config  # noqa: E402
from src.data_loader import sinif_indekslerini_oku, test_yukle  # noqa: E402


def _tahminleri_topla(model, test_ds):
    """Test kumesi uzerinde gercek ve tahmin edilen etiketleri toplar."""
    y_true = []
    y_pred = []
    for goruntuler, etiketler in test_ds:
        olasiliklar = model.predict(goruntuler, verbose=0)
        y_pred.extend(np.argmax(olasiliklar, axis=1).tolist())
        y_true.extend(etiketler.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def confusion_matrix_ciz(y_true, y_pred, sinif_isimleri):
    """Karmasiklik matrisini (confusion matrix) cizip kaydeder."""
    config.klasorleri_hazirla()
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Karmasiklik Matrisi (Confusion Matrix)")
    plt.colorbar()

    isaretler = np.arange(len(sinif_isimleri))
    plt.xticks(isaretler, sinif_isimleri, rotation=45, ha="right")
    plt.yticks(isaretler, sinif_isimleri)

    # Hucre degerlerini yaz.
    esik = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > esik else "black",
            )

    plt.ylabel("Gercek Sinif")
    plt.xlabel("Tahmin Edilen Sinif")
    plt.tight_layout()
    plt.savefig(config.CIKTI_DIZIN / "confusion_matrix.png", dpi=150)
    plt.close()


def degerlendir():
    """Tum degerlendirme akisini calistirir."""
    config.klasorleri_hazirla()

    if not config.MODEL_DOSYA.exists():
        raise FileNotFoundError(
            "Egitilmis model bulunamadi. Once 'python train.py' calistirin."
        )

    print("[1/4] Model ve test verisi yukleniyor...")
    model = tf.keras.models.load_model(str(config.MODEL_DOSYA))
    test_ds, test_sinif_isimleri = test_yukle()
    sinif_isimleri = sinif_indekslerini_oku()
    if list(test_sinif_isimleri) != list(sinif_isimleri):
        raise ValueError(
            "Test sinif sirasi ile kaydedilen class_indices.json uyusmuyor. "
            "Lutfen modeli mevcut veri setiyle yeniden egitin (python train.py)."
        )

    print("[2/4] Test seti uzerinde tahminler yapiliyor...")
    y_true, y_pred = _tahminleri_topla(model, test_ds)

    print("[3/4] Metrikler hesaplaniyor...")
    accuracy = accuracy_score(y_true, y_pred)
    metrikler = {
        "accuracy": float(accuracy),
        "precision_makro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_makro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_makro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "precision_agirlikli": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_agirlikli": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_agirlikli": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }

    rapor_metni = classification_report(
        y_true, y_pred, target_names=sinif_isimleri, zero_division=0
    )

    print("[4/4] Sonuclar kaydediliyor...")
    # Classification report metnini kaydet.
    with open(
        config.RAPOR_DIZIN / "classification_report.txt", "w", encoding="utf-8"
    ) as dosya:
        dosya.write("MEYVE TAZELIK SINIFLANDIRMA - CLASSIFICATION REPORT\n")
        dosya.write("=" * 55 + "\n\n")
        dosya.write(rapor_metni)
        dosya.write("\n\nGenel Dogruluk (Accuracy): {:.4f}\n".format(accuracy))

    # Metrik ozetini JSON olarak kaydet (PDF uretimi de bunu kullanir).
    with open(config.RAPOR_DIZIN / "metrikler.json", "w", encoding="utf-8") as dosya:
        json.dump(metrikler, dosya, ensure_ascii=False, indent=2)

    confusion_matrix_ciz(y_true, y_pred, sinif_isimleri)

    print("\nDegerlendirme tamamlandi.")
    print(rapor_metni)
    print(f"Genel Dogruluk: {accuracy:.4f}")
    print(f"Ciktilar: {config.RAPOR_DIZIN} ve {config.CIKTI_DIZIN}")


if __name__ == "__main__":
    degerlendir()
