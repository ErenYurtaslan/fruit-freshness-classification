# -*- coding: utf-8 -*-
"""Mendeley Fresh and Rotten Fruits veri setini indirir ve acar.

DOI: 10.17632/bdd69gyhv8.1 (CC BY 4.0)
Hedef: temp_dataset/ altina orijinal goruntu klasor yapisi.

Kullanim:
    python download_mendeley.py
"""

import zipfile
from pathlib import Path

import requests

PROJE_KOK = Path(__file__).resolve().parent
HEDEF_ZIP = PROJE_KOK / "temp_dataset" / "mendeley_fruits.zip"
CIKTI_KOK = PROJE_KOK / "temp_dataset"

# Mendeley'nin bilinen S3 cache URL kalibi (dataset DOI'sine gore).
ADAY_URL_LISTESI = [
    # ia-frutas (GitHub) README'de belirtilen Mendeley dogrudan indirme linki.
    "https://data.mendeley.com/public-files/datasets/bdd69gyhv8/files/"
    "ccd1f142-03b2-473a-8c78-78920e63b8bd/file_downloaded",
    "https://md-datasets-cache-apache-aws-production.s3.eu-west-1.amazonaws.com/bdd69gyhv8-1.zip",
    "https://md-datasets-public-files-production.s3.eu-west-1.amazonaws.com/bdd69gyhv8-1.zip",
]


def indir(url: str, hedef: Path) -> bool:
    """URL'den zip indirmeyi dener; basariliysa True."""
    print(f"Deneniyor: {url}")
    try:
        with requests.get(url, stream=True, timeout=120) as yanit:
            if yanit.status_code != 200:
                print(f"  HTTP {yanit.status_code}")
                return False
            icerik_tipi = yanit.headers.get("Content-Type", "")
            if "html" in icerik_tipi.lower():
                print("  HTML dondu (dogrudan indirme basarisiz)")
                return False
            hedef.parent.mkdir(parents=True, exist_ok=True)
            toplam = int(yanit.headers.get("content-length", 0))
            yazilan = 0
            with open(hedef, "wb") as f:
                for parca in yanit.iter_content(chunk_size=1024 * 1024):
                    if parca:
                        f.write(parca)
                        yazilan += len(parca)
                        if toplam:
                            print(f"\r  Indiriliyor: %{100 * yazilan / toplam:.1f}", end="")
            print(f"\n  Kaydedildi: {hedef} ({yazilan / 1024 / 1024:.1f} MB)")
            return yazilan > 1_000_000
    except Exception as exc:
        print(f"  Hata: {exc}")
        return False


def zip_ac(zip_yolu: Path, hedef: Path) -> None:
    """Zip dosyasini hedef klasore acar."""
    print(f"Aciliyor: {zip_yolu}")
    with zipfile.ZipFile(zip_yolu, "r") as zf:
        zf.extractall(hedef)
    print(f"Acildi: {hedef}")


def kaynak_klasorunu_bul(kok: Path) -> Path:
    """original/Original Image yapisini bulur."""
    for aday in [
        kok / "original" / "Original Image",
        kok / "Original Image",
        kok / "original",
    ]:
        if aday.exists() and any(aday.iterdir()):
            return aday
    # Derin arama
    for p in kok.rglob("Original Image"):
        if p.is_dir() and any(p.iterdir()):
            return p
    for p in kok.rglob("FreshApple"):
        if p.is_dir():
            return p.parent
    raise FileNotFoundError(
        "Mendeley klasor yapisi bulunamadi. temp_dataset icinde "
        "'original/Original Image' bekleniyor."
    )


def main():
    if (CIKTI_KOK / "original" / "Original Image" / "FreshApple").exists():
        print("Mendeley verisi zaten mevcut, indirme atlandi.")
        return

    basarili = False
    for url in ADAY_URL_LISTESI:
        if indir(url, HEDEF_ZIP):
            basarili = True
            break

    if not basarili:
        raise RuntimeError(
            "Otomatik indirme basarisiz. Manuel adim:\n"
            "1) https://data.mendeley.com/datasets/bdd69gyhv8/1 adresinden 'Download All'\n"
            f"2) Zip'i su konuma koy: {HEDEF_ZIP}\n"
            "3) python download_mendeley.py tekrar calistir"
        )

    zip_ac(HEDEF_ZIP, CIKTI_KOK)
    kaynak = kaynak_klasorunu_bul(CIKTI_KOK)
    print(f"Kaynak klasor: {kaynak}")
    siniflar = sorted(d.name for d in kaynak.iterdir() if d.is_dir())
    print(f"Sinif sayisi: {len(siniflar)} -> {siniflar[:4]}...")


if __name__ == "__main__":
    main()
