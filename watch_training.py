# -*- coding: utf-8 -*-
"""Egitim log dosyasini canli izler.

Kullanim:
    python watch_training.py --follow          # satir satir anlik akis (tail -f)
    python watch_training.py --follow --tail 30
    python watch_training.py                   # ozet panel (2 sn yenileme)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+)")
BATCH_RE = re.compile(
    r"(\d+)/(\d+).*?accuracy: ([0-9.]+).*?loss: ([0-9.]+)"
    r"(?:.*?val_accuracy: ([0-9.]+).*?val_loss: ([0-9.]+))?"
)
ASAMA_RE = re.compile(r"\[(\d+)/6\] ASAMA (\d+)")
VAL_IMPROVED_RE = re.compile(
    r"val_accuracy improved from ([0-9.inf]+) to ([0-9.]+)"
)


def _temizle(metin: str) -> str:
    return ANSI_RE.sub("", metin)


def _cubuk(tamamlanan: int, toplam: int, genislik: int = 36) -> str:
    if toplam <= 0:
        return "[" + " " * genislik + "]"
    dolu = min(genislik, int(genislik * tamamlanan / toplam))
    return "[" + "#" * dolu + "-" * (genislik - dolu) + "]"


def _ozet_satirlari(log: Path) -> list[str]:
    if not log.exists():
        return [f"Log bekleniyor: {log}"]

    metin = _temizle(log.read_text(encoding="utf-8", errors="replace"))
    satirlar = [s.strip() for s in metin.splitlines() if s.strip()]

    asama = "?"
    epoch_cur = epoch_max = 0
    batch_cur = batch_max = 0
    acc = loss = val_acc = val_loss = None
    son_val = None
    tamamlandi = False

    for satir in satirlar:
        if "Egitim tamamlandi" in satir:
            tamamlandi = True
        m = ASAMA_RE.search(satir)
        if m:
            asama = m.group(2)
        m = EPOCH_RE.search(satir)
        if m:
            epoch_cur, epoch_max = int(m.group(1)), int(m.group(2))
        m = VAL_IMPROVED_RE.search(satir)
        if m:
            son_val = float(m.group(2))
        m = BATCH_RE.search(satir)
        if m:
            batch_cur, batch_max = int(m.group(1)), int(m.group(2))
            acc = float(m.group(3))
            loss = float(m.group(4))
            if m.group(5):
                val_acc = float(m.group(5))
                val_loss = float(m.group(6))

    cikti = [
        "=== Meyve Tazelik Egitimi — Canli Ozet ===",
        f"Log: {log}",
        "",
        f"Asama      : {asama} / 2  (toplam adim 6)",
        f"Epoch       : {epoch_cur}/{epoch_max or '?'}",
        f"Batch       : {batch_cur}/{batch_max or '?'}  {_cubuk(batch_cur, batch_max)}",
    ]
    if acc is not None:
        cikti.append(f"Train acc   : {acc * 100:.2f}%   loss: {loss:.4f}")
    if val_acc is not None:
        cikti.append(
            f"Val acc     : {val_acc * 100:.2f}%   val_loss: {val_loss:.4f}"
        )
    elif son_val is not None:
        cikti.append(f"Son val acc : {son_val * 100:.2f}%")
    if tamamlandi:
        cikti += ["", "Durum       : TAMAMLANDI"]
    else:
        cikti += [
            "",
            "Yenilemek icin bu pencereyi acik birakin (Ctrl+C ile cikis).",
            "Ham log: Get-Content outputs/train_log_16class.txt -Wait -Tail 20",
        ]
    return cikti


def _satir_satir_izle(log: Path, tail: int) -> None:
    """Log dosyasini tail -f gibi satir satir yazdirir."""
    while not log.exists():
        print(f"Log bekleniyor: {log}", flush=True)
        time.sleep(1)

    with log.open(encoding="utf-8", errors="replace") as dosya:
        satirlar = dosya.readlines()
        if tail > 0 and len(satirlar) > tail:
            satirlar = satirlar[-tail:]
        for satir in satirlar:
            print(_temizle(satir).rstrip(), flush=True)

        while True:
            satir = dosya.readline()
            if satir:
                print(_temizle(satir).rstrip(), flush=True)
            else:
                time.sleep(0.3)


def main() -> None:
    ayristirici = argparse.ArgumentParser(description="Egitim log izleyici")
    ayristirici.add_argument(
        "log",
        nargs="?",
        default="outputs/train_log_16class.txt",
        help="Izlenen log dosyasi",
    )
    ayristirici.add_argument(
        "--follow", "-f",
        action="store_true",
        help="Satir satir anlik akis (tail -f)",
    )
    ayristirici.add_argument(
        "--tail",
        type=int,
        default=15,
        help="--follow ile baslamadan once gosterilecek son satir sayisi",
    )
    ayristirici.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Ozet modunda yenileme araligi (saniye)",
    )
    args = ayristirici.parse_args()
    log = Path(args.log)

    try:
        if args.follow:
            print(f"=== Canli log: {log} (Ctrl+C ile cikis) ===", flush=True)
            _satir_satir_izle(log, tail=args.tail)
            return

        while True:
            sys.stdout.write("\033[H\033[J")
            for satir in _ozet_satirlari(log):
                print(satir)
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nIzleme durduruldu.")


if __name__ == "__main__":
    main()
