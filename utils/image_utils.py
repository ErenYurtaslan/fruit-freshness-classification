"""Arayuzde goruntu ve etiket formatlama yardimcilari."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


def yola_gore_pixmap_olustur(goruntu_yolu: str, hedef_boyut) -> QPixmap:
    """Diskteki goruntuyu hedef boyuta oran koruyarak olcekler."""
    pixmap = QPixmap(goruntu_yolu)
    return pixmap.scaled(
        hedef_boyut,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )


def diziye_gore_pixmap_olustur(goruntu_dizisi: np.ndarray, hedef_boyut) -> QPixmap:
    """RGB uint8 numpy dizisini QPixmap'e cevirir."""
    dizi = np.ascontiguousarray(goruntu_dizisi)
    yukseklik, genislik, kanal = dizi.shape
    satir_bayt = kanal * genislik
    qimg = QImage(dizi.data, genislik, yukseklik, satir_bayt, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        hedef_boyut,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )


def sinif_adini_kullanici_dostu_yap(sinif_adi: str) -> str:
    """Model sinif adini gorunur ve standart etikete cevirir.

    Ornek:
    - freshbanana -> Fresh Banana
    - rottenapples -> Rotten Apple
    """
    temiz = sinif_adi.lower().strip()
    durum = "Fresh" if temiz.startswith("fresh") else "Rotten"

    # fresh/rotten on ekini atip meyve tipini elde et.
    govde = re.sub(r"^(fresh|rotten)", "", temiz)
    # Cogu etiket cogul, tekile indir.
    govde = govde[:-1] if govde.endswith("s") else govde
    ozel = {"gape": "Grape", "grape": "Grape", "jujube": "Jujube"}
    if govde in ozel:
        meyve = ozel[govde]
    else:
        meyve = govde.capitalize() if govde else "Fruit"
    return f"{durum} {meyve}"


def genel_durumu_uret(sinif_adi: str) -> str:
    """Sinif etiketinden genel tazelik durumunu cikarir."""
    return "TAZE" if sinif_adi.lower().startswith("fresh") else "ÇÜRÜK"


def meyve_turunu_uret(sinif_adi: str) -> str:
    """Sinif adindan meyve turunu uretir."""
    temiz = sinif_adi.lower()
    if "apple" in temiz:
        return "Elma"
    if "banana" in temiz:
        return "Muz"
    if "orange" in temiz:
        return "Portakal"
    if "grape" in temiz or "gape" in temiz:
        return "Üzüm"
    if "guava" in temiz:
        return "Guava"
    if "jujube" in temiz:
        return "Hünnap"
    if "pomegranate" in temiz:
        return "Nar"
    if "strawberry" in temiz:
        return "Çilek"
    return "Bilinmiyor"


def dosya_adini_kisa_goster(goruntu_yolu: str) -> str:
    """Uzun dosya adlarini arayuzde okunur gostermek icin kisaltir."""
    ad = Path(goruntu_yolu).name
    if len(ad) <= 48:
        return ad
    return f"{ad[:22]}...{ad[-20:]}"
