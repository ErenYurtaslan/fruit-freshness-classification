# -*- coding: utf-8 -*-
"""Mendeley veri seti guclendirme ve yeniden egitim pipeline'i.

Tum adimlari sirayla otomatik calistirir; kullanicinin Run'a basmasina
gerek kalmadan indirme, entegrasyon, egitim, kalibrasyon ve raporlama
yapilir.

Kullanim:
    python run_mendeley_pipeline.py              # tam pipeline
    python run_mendeley_pipeline.py --skip-train # egitim atla (model hazir)
    python run_mendeley_pipeline.py --ood-only   # yalnizca ood_eval + kalibrasyon
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _calistir(komut: list[str], aciklama: str) -> None:
    print(f"\n=== {aciklama} ===")
    print(" ".join(komut))
    sonuc = subprocess.run(komut, check=False)
    if sonuc.returncode != 0:
        raise SystemExit(f"Hata: {aciklama} basarisiz (kod={sonuc.returncode})")


def main() -> None:
    kok = Path(__file__).resolve().parent
    py = sys.executable

    ayristirici = argparse.ArgumentParser(description="Mendeley pipeline")
    ayristirici.add_argument(
        "--skip-train",
        action="store_true",
        help="train.py ve evaluate.py atlanir (model zaten egitilmis).",
    )
    ayristirici.add_argument(
        "--ood-only",
        action="store_true",
        help="Yalnizca ood_eval + OOD kalibrasyon + dogrulama.",
    )
    ayristirici.add_argument(
        "--skip-download",
        action="store_true",
        help="Mendeley zip indirmeyi atla.",
    )
    args = ayristirici.parse_args()

    if not args.skip_download and not args.ood_only:
        zip_dosya = kok / "temp_dataset" / "mendeley_fruits.zip"
        if not zip_dosya.exists():
            _calistir([py, "download_mendeley.py"], "Mendeley indirme")

    if args.ood_only:
        _calistir(
            [py, "integrate_mendeley.py", "--ood-only"],
            "OOD eval seti olusturma",
        )
    else:
        _calistir(
            [py, "integrate_mendeley.py", "--full-augmented", "--cleanup-temp"],
            "Tam Augmented Image birlestirme",
        )
        if not args.skip_train:
            _calistir([py, "train.py"], "Model egitimi")
            _calistir([py, "evaluate.py"], "Model degerlendirme")

    ood_stats = kok / "models" / "ood_stats.json"
    if ood_stats.exists():
        ood_stats.unlink()
        print("Eski ood_stats.json silindi.")

    _calistir([py, "calibrate_temperature.py"], "Temperature kalibrasyonu")
    _calistir([py, "calibrate_ood.py"], "OOD esik kalibrasyonu")
    _calistir([py, "compare_ood.py"], "OOD karsilastirma raporu")
    _calistir([py, "validate_mendeley.py"], "Hedef davranis dogrulama")
    _calistir([py, "generate_report.py"], "Akademik PDF guncelleme")

    print("\n=== Pipeline tamamlandi ===")


if __name__ == "__main__":
    main()
