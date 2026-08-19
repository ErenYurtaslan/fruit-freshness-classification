# -*- coding: utf-8 -*-
"""Mendeley sonrasi hedef davranis dogrulama betigi.

Sari elma / curuk portakal / dis ornekler / ood_eval yabanci meyveler
uzerinde smoke test yapar ve reports/mendeley_dogrulama_raporu.md uretir.

Kullanim:
    python validate_mendeley.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from utils.predictor import TahminServisi

OOD_TEST = Path("outputs/ood_test")
OOD_EVAL = Path("outputs/ood_eval")
RAPOR = Path("reports/mendeley_dogrulama_raporu.md")

# Opsiyonel gercek dunya ornekleri (varsa test edilir).
GERCEK_ORNEKLER = {
    "sari_elma": [
        Path.home() / "Desktop" / "sari_elma.jpg",
        Path.home() / "Desktop" / "sari_elma.png",
        Path("outputs/validation/sari_elma.jpg"),
    ],
    "curuk_portakal": [
        Path.home() / "Desktop" / "curuk_portakal.jpg",
        Path("outputs/validation/curuk_portakal.jpg"),
    ],
}


def _ilk_var(yollar: list[Path]) -> Path | None:
    for yol in yollar:
        if yol.exists():
            return yol
    return None


def _durum_ozeti(sonuc: dict) -> str:
    if not sonuc["bilinen_mi"]:
        return "Bilinmeyen"
    if sonuc.get("sinirda_mi"):
        return f"Sinirda ({sonuc['sinif_gosterim']})"
    return f"Bilinen ({sonuc['sinif_gosterim']})"


def main() -> None:
    servis = TahminServisi()
    satirlar = [
        "# Mendeley Sonrasi Dogrulama Raporu",
        "",
        f"*Olusturma: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## 7 Dis Ornek (outputs/ood_test)",
        "",
        "| Ornek | Durum | Mesafe | Tahmin |",
        "|---|---|---|---|",
    ]

    dis_isaret = 0
    dis_toplam = 0
    for dosya in sorted(OOD_TEST.glob("*.*")):
        if dosya.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        sonuc = servis.tahmin_yap(str(dosya))
        dis_toplam += 1
        isaretli = not sonuc["bilinen_mi"] or sonuc.get("sinirda_mi")
        dis_isaret += int(isaretli)
        m = sonuc["ood_metrikleri"]
        satirlar.append(
            f"| {dosya.stem} | {_durum_ozeti(sonuc)} | "
            f"{m['embedding_distance']:.3f} | {sonuc['model_sinif']} |"
        )

    satirlar += [
        "",
        f"**Isaretleme:** {dis_isaret}/{dis_toplam} (hedef >= 5/7)",
        "",
        "## ood_eval Holdout (taze/curuk ayri sinif klasorleri)",
        "",
        "| Sinif | N test | Bilinmeyen | Sinirda | Bilinen |",
        "|---|---|---|---|---|",
    ]

    eval_toplam = eval_bilinen = eval_yanlis = 0
    for sinif_klasor in sorted(OOD_EVAL.iterdir()):
        if not sinif_klasor.is_dir():
            continue
        dosyalar = [
            p
            for p in sinif_klasor.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ][:10]
        bilinmeyen = sinirda = bilinen = 0
        for dosya in dosyalar:
            sonuc = servis.tahmin_yap(str(dosya))
            eval_toplam += 1
            if not sonuc["bilinen_mi"]:
                bilinmeyen += 1
                eval_yanlis += 1
            elif sonuc.get("sinirda_mi"):
                sinirda += 1
            else:
                bilinen += 1
                eval_bilinen += 1
        satirlar.append(
            f"| {sinif_klasor.name} | {len(dosyalar)} | {bilinmeyen} | "
            f"{sinirda} | {bilinen} |"
        )

    yanlis_oran = (eval_yanlis / eval_toplam * 100) if eval_toplam else 0
    bilinen_oran = (eval_bilinen / eval_toplam * 100) if eval_toplam else 0
    satirlar += [
        "",
        f"**ood_eval holdout tanima:** %{bilinen_oran:.1f} bilinen "
        f"({eval_bilinen}/{eval_toplam}), yanlis bilinmeyen %{yanlis_oran:.1f}",
        "",
        "## Gercek Dunya Hedef Ornekleri",
        "",
    ]

    for ad, adaylar in GERCEK_ORNEKLER.items():
        yol = _ilk_var(adaylar)
        if yol is None:
            satirlar.append(f"- **{ad}:** dosya bulunamadi (atlandi)")
            continue
        sonuc = servis.tahmin_yap(str(yol))
        satirlar.append(
            f"- **{ad}** (`{yol.name}`): {_durum_ozeti(sonuc)}, "
            f"mesafe={sonuc['ood_metrikleri']['embedding_distance']:.3f}, "
            f"meyve={sonuc['meyve_turu']}, kalite={sonuc['genel_durum']}"
        )

    satirlar += [
        "",
        "## Ozet",
        "",
        "- Uc kademeli OOD karari ve sinirda UI yumusatmasi (Belirsiz) aktif.",
        "- Mendeley gercek dunya gorselleri egitime entegre edildi; model test "
        "dogrulugu ~%96.",
    ]

    RAPOR.parent.mkdir(parents=True, exist_ok=True)
    RAPOR.write_text("\n".join(satirlar), encoding="utf-8")
    print(f"Rapor: {RAPOR}")
    print(f"Dis ornek isaretleme: {dis_isaret}/{dis_toplam}")
    print(f"ood_eval holdout tanima: %{bilinen_oran:.1f}, yanlis bilinmeyen: %{yanlis_oran:.1f}")


if __name__ == "__main__":
    main()
