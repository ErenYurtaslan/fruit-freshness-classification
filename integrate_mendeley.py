# -*- coding: utf-8 -*-
"""Mendeley "Fresh and Rotten Fruits Dataset" entegrasyon scripti.

Kaynak: temp_dataset/Augmented Image (veya Original Image varsa oncelikli).

Tum 16 sinif (11 meyve x taze/curuk) dataset/train|test altina %80/20 eklenir:
  elma, muz, portakal, uzum, guava, hunnap, nar, cilek.
- --full-augmented: Augmented Image'deki TUM gorselleri mendeley_aug_* onekiyle ekler.
- outputs/ood_eval/: harici OOD dogrulama ornekleri (egitimden ayri kucuk orneklem).
"""

import random
from pathlib import Path

from PIL import Image

def _augmented_kaynak() -> Path:
    """Augmented Image klasorunu dondurur."""
    aday = Path("temp_dataset/Augmented Image")
    if aday.exists() and any(aday.iterdir()):
        return aday
    raise FileNotFoundError(
        "temp_dataset/Augmented Image bulunamadi. Once download_mendeley.py calistirin."
    )


def _kaynak_klasorunu_bul() -> Path:
    """Orijinal klasoru tercih eder; yoksa artirilmis sete duser."""
    adaylar = [
        Path("temp_dataset/original/Original Image"),
        Path("temp_dataset/Augmented Image"),
        Path("temp_dataset/Original Image"),
    ]
    for aday in adaylar:
        if aday.exists() and any(aday.iterdir()):
            return aday
    raise FileNotFoundError(
        "Mendeley kaynak klasoru bulunamadi. Once python download_mendeley.py calistirin."
    )
DATASET = Path("dataset")
OOD_EVAL = Path("outputs/ood_eval")

# Mendeley sinif adi -> dataset sinif klasoru (16 sinif = 8 meyve x taze/curuk)
EGITIM_ESLEME = {
    "FreshApple": "freshapples",
    "RottenApple": "rottenapples",
    "FreshBanana": "freshbanana",
    "RottenBanana": "rottenbanana",
    "FreshOrange": "freshoranges",
    "RottenOrange": "rottenoranges",
    "FreshGrape": "freshgrapes",
    "RottenGrape": "rottengapes",
    "FreshGuava": "freshguava",
    "RottenGuava": "rottenguava",
    "FreshJujube": "freshjujube",
    "RottenJujube": "rottenjujube",
    "FreshPomegranate": "freshpomegranate",
    "RottenPomegranate": "rottenpomegranate",
    "FreshStrawberry": "freshstrawberry",
    "RottenStrawberry": "rottenstrawberry",
}

# OOD holdout: egitimle AYNI sinif adlari (taze/curuk ayri klasorler)
OOD_HOLDOUT_SINIFLAR = {
    mendeley: hedef
    for mendeley, hedef in EGITIM_ESLEME.items()
    if hedef
    not in {
        "freshapples", "rottenapples",
        "freshbanana", "rottenbanana",
        "freshoranges", "rottenoranges",
    }
}

TEST_ORANI = 0.20
OOD_ORNEK_SAYISI = 20  # her Mendeley sinifindan (taze/curuk ayri) alinacak ornek
MAKS_KENAR = 512
RASTGELE_TOHUM = 42


def kucult_ve_kaydet(kaynak: Path, hedef: Path) -> None:
    """Gorseli uzun kenari MAKS_KENAR olacak sekilde kucultup JPEG kaydeder."""
    with Image.open(kaynak) as im:
        im = im.convert("RGB")
        im.thumbnail((MAKS_KENAR, MAKS_KENAR), Image.LANCZOS)
        im.save(hedef, "JPEG", quality=90)


def egitim_setine_ekle(kaynak: Path, onek: str = "mendeley_") -> tuple[int, int]:
    """Mendeley gorsellerini train/test'e kopyalar. (train, test) adet dondurur."""
    rng = random.Random(RASTGELE_TOHUM)
    train_toplam = test_toplam = 0
    for mendeley_sinif, hedef_sinif in EGITIM_ESLEME.items():
        klasor = kaynak / mendeley_sinif
        if not klasor.exists():
            print(f"  ATLA: {mendeley_sinif} klasoru yok")
            continue
        dosyalar = sorted(
            p for p in klasor.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        rng.shuffle(dosyalar)
        test_adet = int(len(dosyalar) * TEST_ORANI)
        bolumler = {
            "test": dosyalar[:test_adet],
            "train": dosyalar[test_adet:],
        }
        sinif_train = sinif_test = 0
        for bolum, liste in bolumler.items():
            hedef_dizin = DATASET / bolum / hedef_sinif
            hedef_dizin.mkdir(parents=True, exist_ok=True)
            for dosya in liste:
                hedef = hedef_dizin / f"{onek}{dosya.stem}.jpg"
                if hedef.exists():
                    continue
                kucult_ve_kaydet(dosya, hedef)
                if bolum == "train":
                    sinif_train += 1
                else:
                    sinif_test += 1
        train_toplam += sinif_train
        test_toplam += sinif_test
        print(
            f"{mendeley_sinif} -> {hedef_sinif}: "
            f"+train={sinif_train} +test={sinif_test} "
            f"(kaynak={len(dosyalar)})"
        )
    return train_toplam, test_toplam


def tam_augmented_birlestir() -> tuple[int, int]:
    """Augmented Image'deki 16 sinifin tum gorsellerini dataset'e ekler."""
    kaynak = _augmented_kaynak()
    print(f"Tam birlestirme (16 sinif): {kaynak}")
    return egitim_setine_ekle(kaynak, onek="mendeley_aug_")


def ood_setini_olustur(kaynak: Path) -> None:
    """Holdout ornekleri dataset ile ayni sinif klasor yapisinda olusturur."""
    import shutil

    rng = random.Random(RASTGELE_TOHUM)
    if OOD_EVAL.exists():
        shutil.rmtree(OOD_EVAL)
    OOD_EVAL.mkdir(parents=True, exist_ok=True)

    for mendeley_sinif, hedef_sinif in OOD_HOLDOUT_SINIFLAR.items():
        klasor = kaynak / mendeley_sinif
        if not klasor.exists():
            print(f"  OOD ATLA: {mendeley_sinif} yok")
            continue
        dosyalar = sorted(
            p for p in klasor.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        secilen = rng.sample(dosyalar, min(OOD_ORNEK_SAYISI, len(dosyalar)))
        hedef_dizin = OOD_EVAL / hedef_sinif
        hedef_dizin.mkdir(parents=True, exist_ok=True)
        for dosya in secilen:
            hedef = hedef_dizin / f"holdout_{dosya.stem}.jpg"
            kucult_ve_kaydet(dosya, hedef)
        print(f"ood_eval/{hedef_sinif}: {len(secilen)} gorsel")


if __name__ == "__main__":
    import argparse

    ayristirici = argparse.ArgumentParser(description="Mendeley veri seti entegrasyonu")
    ayristirici.add_argument(
        "--ood-only",
        action="store_true",
        help="Yalnizca outputs/ood_eval setini olustur (train/test'e dokunma).",
    )
    ayristirici.add_argument(
        "--full-augmented",
        action="store_true",
        help="Augmented Image'deki tum 16 sinifi dataset'e ekle.",
    )
    ayristirici.add_argument(
        "--force-train",
        action="store_true",
        help="Train/test'e tekrar ekle (varsayilan: mevcut mendeley_* varsa atla).",
    )
    ayristirici.add_argument(
        "--cleanup-temp",
        action="store_true",
        help="Birlestirme sonrasi temp_dataset klasorunu sil.",
    )
    args = ayristirici.parse_args()

    kaynak = _kaynak_klasorunu_bul()
    print(f"=== Mendeley entegrasyonu basliyor (kaynak: {kaynak}) ===")

    if args.ood_only:
        print("Train/test atlandi (--ood-only).")
    elif args.full_augmented:
        tr, te = tam_augmented_birlestir()
        print(f"Toplam eklenen: train={tr}, test={te}")
    else:
        mevcut_mendeley = list((DATASET / "train" / "freshapples").glob("mendeley_*"))
        train_ekle = args.force_train or not mevcut_mendeley
        if train_ekle:
            tr, te = egitim_setine_ekle(kaynak)
            print(f"Toplam eklenen: train={tr}, test={te}")
        else:
            print("Train/test zaten entegre (mendeley_* mevcut), atlandi.")
            print("Tam birlestirme icin: python integrate_mendeley.py --full-augmented")

    ood_setini_olustur(kaynak)

    if args.cleanup_temp:
        import shutil
        temp = Path("temp_dataset")
        if temp.exists():
            shutil.rmtree(temp)
            print("temp_dataset silindi.")

    print("=== Tamamlandi ===")
