"""OOD esik kalibrasyon betigi.

Mesafe-oncelikli hiyerarsik kararin esiklerini uc kisit altinda dogrular:
1. Bilinen test orneklerinde yanlis "bilinmeyen" orani <= %2 olmali.
2. 7 dis ornekte isaretleme (bilinmeyen + sinirda) orani >= 5/7 olmali.
3. outputs/ood_eval holdout setinde (egitimdeki yeni siniflar, taze/curuk ayri)
   yanlis "bilinmeyen" orani <= %10 olmali.

Betik iki boyutlu bir esik izgarasi (T_dusuk x T_yuksek) tarar, her aday
icin uc metrigi hesaplar, mevcut config degerlerinin kisitlari saglayip
saglamadigini raporlar ve mesafe dagiliminin yuzdelik ozetini cikarir.

Ciktilar:
- reports/ood_kalibrasyon_raporu.md
- outputs/ood_analiz/kalibrasyon_metrikleri.npz (karsilastirma icin yeniden kullanilir)

Kullanim:
    python calibrate_ood.py
"""

import json
from datetime import datetime

import numpy as np
import tensorflow as tf

from src import config
from src.utils import goruntu_yukle_ve_hazirla
from utils.ood_detector import OODAlgilayici

ANALIZ_DIZIN = config.CIKTI_DIZIN / "ood_analiz"
RAPOR_DOSYA = config.RAPOR_DIZIN / "ood_kalibrasyon_raporu.md"
OOD_TEST_DIZIN = config.CIKTI_DIZIN / "ood_test"
OOD_EVAL_DIZIN = config.CIKTI_DIZIN / "ood_eval"

# Sinif basina ornek: 6 sinif x 84 = 504 bilinen test ornegi.
SINIF_BASI_ORNEK = 84

DIS_ORNEK_ADLARI = {
    "ejder_meyvesi": "Ejder Meyvesi",
    "lici": "Liçi",
    "mor_uzum": "Mor Üzüm",
    "mor_erik": "Mor Erik",
    "murdum_erigi": "Mürdüm Eriği",
    "mandalina": "Mandalina",
    "greyfurt": "Greyfurt",
}

from utils.image_utils import sinif_adini_kullanici_dostu_yap


def _sinif_isimlerini_oku():
    esleme = json.loads(config.SINIF_INDEKS_DOSYA.read_text(encoding="utf-8"))
    return [esleme[str(i)] for i in range(len(esleme))]


def _toplu_metrik_olc(model, detektor, yollar):
    """Bir dosya listesi icin mesafe/maxprob/entropy/sinif metriklerini olcer."""
    girdi = detektor._batch_onisle(yollar)
    olasilik = model.predict(girdi, batch_size=32, verbose=0)
    embedding = detektor.embedding_model.predict(girdi, batch_size=32, verbose=0)
    merkezler = detektor.istatistikler["sinif_merkezleri"]
    mesafeler = np.array(
        [detektor._en_yakin_merkez_uzakligi(v, merkezler) for v in embedding]
    )
    en_yakin = np.array(
        [detektor._en_yakin_sinif(v, merkezler) for v in embedding]
    )
    max_prob = np.max(olasilik, axis=1)
    entropy = detektor._normalize_entropy(olasilik)
    tahminler = np.array(
        [detektor.sinif_isimleri[i] for i in np.argmax(olasilik, axis=1)]
    )
    return {
        "mesafeler": mesafeler,
        "max_prob": max_prob,
        "entropy": entropy,
        "tahmin": tahminler,
        "en_yakin": en_yakin,
    }


def bilinen_metrikleri_olc(model, detektor, sinif_isimleri):
    """Bilinen test ornekleri icin mesafe/maxprob/entropy ve dogruluk olcer."""
    yollar, gercek_siniflar = [], []
    for sinif in sinif_isimleri:
        klasor = config.TEST_DIZIN / sinif
        dosyalar = sorted(
            p
            for p in klasor.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )[:SINIF_BASI_ORNEK]
        yollar.extend(dosyalar)
        gercek_siniflar.extend([sinif] * len(dosyalar))

    metrikler = _toplu_metrik_olc(model, detektor, yollar)
    metrikler["gercek"] = np.array(gercek_siniflar)
    return metrikler


def dis_metrikleri_olc(model, detektor):
    """7 dis ornegin metriklerini olcer."""
    sonuclar = []
    for govde, ad in DIS_ORNEK_ADLARI.items():
        eslesen = list(OOD_TEST_DIZIN.glob(f"{govde}.*"))
        if not eslesen:
            continue
        model_girisi, _ = goruntu_yukle_ve_hazirla(str(eslesen[0]))
        olasilik = model.predict(model_girisi, verbose=0)[0]
        ood = detektor.degerlendir(model_girisi, olasilik)
        sonuclar.append(
            {
                "ad": ad,
                "mesafe": ood["embedding_distance"],
                "max_prob": ood["max_prob"],
                "entropy": ood["entropy"],
                "softmax_sinif": detektor.sinif_isimleri[int(np.argmax(olasilik))],
                "en_yakin_sinif": ood["en_yakin_sinif"],
            }
        )
    return sonuclar


def ood_eval_metrikleri_olc(model, detektor):
    """outputs/ood_eval holdout setinin metriklerini sinif bazinda olcer."""
    sonuclar = {}
    if not OOD_EVAL_DIZIN.exists():
        return sonuclar
    for meyve_dizin in sorted(OOD_EVAL_DIZIN.iterdir()):
        if not meyve_dizin.is_dir():
            continue
        yollar = sorted(
            p
            for p in meyve_dizin.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not yollar:
            continue
        sonuclar[meyve_dizin.name] = _toplu_metrik_olc(model, detektor, yollar)
    return sonuclar


def _durumlari_hesapla(metrikler, t_yuksek=None, t_dusuk=None):
    """Bir metrik paketi icin uc kademeli durum dizisi uretir."""
    return np.array(
        [
            OODAlgilayici.hiyerarsik_karar(
                float(d), float(p), float(e),
                softmax_sinif=str(t), en_yakin_sinif=str(y),
                t_yuksek=t_yuksek, t_dusuk=t_dusuk,
            )[0]
            for d, p, e, t, y in zip(
                metrikler["mesafeler"], metrikler["max_prob"],
                metrikler["entropy"], metrikler["tahmin"], metrikler["en_yakin"],
            )
        ]
    )


def _ood_eval_yanlis_bilinmeyen_orani(ood_eval, t_yuksek, t_dusuk):
    """Holdout setinde yanlis bilinmeyen isaretleme oranini dondurur (dusuk olmali)."""
    toplam, yanlis = 0, 0
    for metrikler in ood_eval.values():
        durumlar = _durumlari_hesapla(metrikler, t_yuksek=t_yuksek, t_dusuk=t_dusuk)
        toplam += len(durumlar)
        yanlis += int(np.sum(durumlar == "bilinmeyen"))
    return (yanlis / toplam) if toplam else 0.0


def izgara_tara(bilinen, dis_sonuclar, ood_eval):
    """Iki boyutlu esik izgarasini tarayip kisit metriklerini hesaplar."""
    satirlar = []
    for t_dusuk in (0.305, 0.330, 0.355, 0.380, 0.405):
        for t_yuksek in np.arange(0.380, 0.561, 0.010):
            t_yuksek = round(float(t_yuksek), 3)
            if t_yuksek <= t_dusuk:
                continue
            durumlar = _durumlari_hesapla(
                bilinen, t_yuksek=t_yuksek, t_dusuk=t_dusuk
            )
            yanlis_bilinmeyen_orani = float(np.mean(durumlar == "bilinmeyen"))
            sinirda_orani = float(np.mean(durumlar == "sinirda"))

            kesin_yakalanan = 0
            uyarili = 0
            for s in dis_sonuclar:
                durum, _ = OODAlgilayici.hiyerarsik_karar(
                    s["mesafe"], s["max_prob"], s["entropy"],
                    softmax_sinif=s["softmax_sinif"],
                    en_yakin_sinif=s["en_yakin_sinif"],
                    t_yuksek=t_yuksek, t_dusuk=t_dusuk,
                )
                if durum == "bilinmeyen":
                    kesin_yakalanan += 1
                elif durum == "sinirda":
                    uyarili += 1

            eval_yanlis = _ood_eval_yanlis_bilinmeyen_orani(ood_eval, t_yuksek, t_dusuk)

            satirlar.append(
                {
                    "t_dusuk": t_dusuk,
                    "t_yuksek": t_yuksek,
                    "yanlis_bilinmeyen": yanlis_bilinmeyen_orani,
                    "sinirda": sinirda_orani,
                    "kesin_yakalanan": kesin_yakalanan,
                    "uyarili": uyarili,
                    "eval_yanlis_bilinmeyen": eval_yanlis,
                    "uygun": (
                        yanlis_bilinmeyen_orani <= 0.02
                        and (kesin_yakalanan + uyarili) >= 5
                        and eval_yanlis <= 0.10
                    ),
                }
            )
    return satirlar


def yuzdelik_ozeti(bilinen, dis_sonuclar, ood_eval):
    """Mesafe dagilimlarinin yuzdelik ozetini cikarir."""
    satirlar = []
    b = bilinen["mesafeler"]
    satirlar.append(
        ("Bilinen test", len(b), np.percentile(b, 50), np.percentile(b, 90),
         np.percentile(b, 95), np.percentile(b, 99), np.max(b))
    )
    if dis_sonuclar:
        d = np.array([s["mesafe"] for s in dis_sonuclar])
        satirlar.append(
            ("7 dış örnek", len(d), np.percentile(d, 50), np.percentile(d, 90),
             np.percentile(d, 95), np.percentile(d, 99), np.max(d))
        )
    for meyve, metrikler in ood_eval.items():
        m = metrikler["mesafeler"]
        satirlar.append(
            (f"ood_eval/{sinif_adini_kullanici_dostu_yap(meyve)}", len(m),
             np.percentile(m, 50), np.percentile(m, 90),
             np.percentile(m, 95), np.percentile(m, 99), np.max(m))
        )
    return satirlar


def main():
    config.klasorleri_hazirla()
    ANALIZ_DIZIN.mkdir(parents=True, exist_ok=True)

    print("Model yukleniyor...")
    model = tf.keras.models.load_model(str(config.MODEL_DOSYA))
    sinif_isimleri = _sinif_isimlerini_oku()
    detektor = OODAlgilayici(model, sinif_isimleri)

    print(f"Bilinen test ornekleri olculuyor ({SINIF_BASI_ORNEK}/sinif)...")
    bilinen = bilinen_metrikleri_olc(model, detektor, sinif_isimleri)
    print("Dis ornekler olculuyor...")
    dis_sonuclar = dis_metrikleri_olc(model, detektor)
    print("ood_eval holdout seti olculuyor...")
    ood_eval = ood_eval_metrikleri_olc(model, detektor)

    # Metrikleri karsilastirma betigi icin sakla.
    np.savez(
        ANALIZ_DIZIN / "kalibrasyon_metrikleri.npz",
        mesafeler=bilinen["mesafeler"],
        max_prob=bilinen["max_prob"],
        entropy=bilinen["entropy"],
        gercek=bilinen["gercek"],
        tahmin=bilinen["tahmin"],
        en_yakin=bilinen["en_yakin"],
    )

    print("Esik izgarasi taraniyor...")
    izgara = izgara_tara(bilinen, dis_sonuclar, ood_eval)
    yuzdelikler = yuzdelik_ozeti(bilinen, dis_sonuclar, ood_eval)

    # Mevcut config degerlerinin durumu.
    mevcut = next(
        (
            r
            for r in izgara
            if abs(r["t_yuksek"] - config.OOD_MESAFE_T_YUKSEK) < 1e-9
            and abs(r["t_dusuk"] - config.OOD_MESAFE_T_DUSUK) < 1e-9
        ),
        None,
    )

    # Bilinen orneklerde durum dagilimi ve dogruluk (OOD sonrasi).
    durumlar = _durumlari_hesapla(bilinen)
    maske_gosterilen = durumlar != "bilinmeyen"
    ham_dogruluk = float(np.mean(bilinen["gercek"] == bilinen["tahmin"]))
    kalan_dogruluk = float(
        np.mean(
            (bilinen["gercek"] == bilinen["tahmin"])[maske_gosterilen]
        )
    )
    bilinen_sinirda_orani = float(np.mean(durumlar == "sinirda"))

    satirlar = [
        "# OOD Eşik Kalibrasyon Raporu (Üç Kademeli Karar)",
        "",
        f"*Oluşturma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Karar Modeli",
        "",
        "- `mesafe > T_yüksek` → **Bilinmeyen** (koşulsuz)",
        "- `mesafe < T_düşük` → **Bilinen**",
        "- Gri bölge → softmax sınıfı ile en yakın merkez sınıfı uyuşuyor ve",
        "  güven yüksekse **Sınırda** (tahmin düşük güven uyarısıyla gösterilir),",
        "  aksi halde Bilinmeyen",
        "",
        "## Kısıtlar",
        "",
        "1. Bilinen test örneklerinde yanlış 'bilinmeyen' oranı **≤ %2**",
        "2. 7 dış örnekte işaretleme (kesin bilinmeyen + sınırda uyarısı) **≥ 5/7**",
        "3. ood_eval holdout setinde (egitimdeki yeni siniflar, taze/curuk ayri)",
        "   yanlis bilinmeyen orani **≤ %10**",
        "",
        f"Bilinen örneklem: **{len(bilinen['mesafeler'])}** test görseli "
        f"({SINIF_BASI_ORNEK}/sınıf). Holdout örneklemi: "
        f"**{sum(len(m['mesafeler']) for m in ood_eval.values())}** görsel.",
        "",
        "## Mesafe Dağılımı Yüzdelikleri",
        "",
        "| Küme | N | P50 | P90 | P95 | P99 | Maks |",
        "|---|---|---|---|---|---|---|",
    ]
    for ad, n, p50, p90, p95, p99, mx in yuzdelikler:
        satirlar.append(
            f"| {ad} | {n} | {p50:.3f} | {p90:.3f} | {p95:.3f} "
            f"| {p99:.3f} | {mx:.3f} |"
        )

    satirlar += [
        "",
        "## Eşik Izgarası (uygun adaylar + mevcut config)",
        "",
        "| T_düşük | T_yüksek | Bilinen→yanlış bilinmeyen | Bilinen→sınırda "
        "| Dış: kesin | Dış: sınırda | ood_eval yanlış bilinmeyen | Uygun |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in izgara:
        mevcut_mu = (
            abs(r["t_yuksek"] - config.OOD_MESAFE_T_YUKSEK) < 1e-9
            and abs(r["t_dusuk"] - config.OOD_MESAFE_T_DUSUK) < 1e-9
        )
        # Rapor cok uzamasin diye yalnizca uygun adaylar ve mevcut config yazilir.
        if not (r["uygun"] or mevcut_mu):
            continue
        isaret = " (mevcut)" if mevcut_mu else ""
        satirlar.append(
            f"| {r['t_dusuk']:.3f} | {r['t_yuksek']:.3f}{isaret} "
            f"| %{r['yanlis_bilinmeyen'] * 100:.2f} | %{r['sinirda'] * 100:.2f} "
            f"| {r['kesin_yakalanan']}/7 | {r['uyarili']}/7 "
            f"| %{r['eval_yanlis_bilinmeyen'] * 100:.1f} "
            f"| {'EVET' if r['uygun'] else 'hayır'} |"
        )

    satirlar += [
        "",
        "## Seçilen Eşikler (config)",
        "",
        f"- `OOD_MESAFE_T_YUKSEK = {config.OOD_MESAFE_T_YUKSEK}`",
        f"- `OOD_MESAFE_T_DUSUK = {config.OOD_MESAFE_T_DUSUK}`",
        f"- `OOD_GRI_MAXPROB_ESIK = {config.OOD_GRI_MAXPROB_ESIK}`",
        f"- `OOD_GRI_ENTROPY_ESIK = {config.OOD_GRI_ENTROPY_ESIK}`",
        "",
    ]
    if mevcut:
        satirlar += [
            f"- Bilinen örneklerde yanlış bilinmeyen: **%{mevcut['yanlis_bilinmeyen'] * 100:.2f}** (hedef ≤ %2)",
            f"- Bilinen örneklerde sınırda uyarısı: **%{mevcut['sinirda'] * 100:.2f}**",
            f"- Dış örnek: **{mevcut['kesin_yakalanan']}/7 kesin bilinmeyen + "
            f"{mevcut['uyarili']}/7 sınırda uyarısı** (toplam işaretleme "
            f"{mevcut['kesin_yakalanan'] + mevcut['uyarili']}/7, hedef ≥ 5/7)",
            f"- ood_eval holdout yanlış bilinmeyen: **%{mevcut['eval_yanlis_bilinmeyen'] * 100:.1f}** (hedef ≤ %10)",
            f"- Kısıtlar: **{'SAĞLANDI' if mevcut['uygun'] else 'SAĞLANAMADI'}**",
        ]
    satirlar += [
        "",
        "## Holdout Seti Ayrıntısı (mevcut eşiklerle)",
        "",
        "| Sınıf | N | Bilinmeyen | Sınırda | Bilinen |",
        "|---|---|---|---|---|",
    ]
    for meyve, metrikler in ood_eval.items():
        d = _durumlari_hesapla(metrikler)
        n = len(d)
        satirlar.append(
            f"| {sinif_adini_kullanici_dostu_yap(meyve)} | {n} "
            f"| {int(np.sum(d == 'bilinmeyen'))} "
            f"| {int(np.sum(d == 'sinirda'))} "
            f"| {int(np.sum(d == 'bilinen'))} |"
        )

    satirlar += [
        "",
        "## Doğruluk Etkisi",
        "",
        f"- Örneklemde ham model doğruluğu (OOD yok): **%{ham_dogruluk * 100:.2f}**",
        f"- Gösterilen (bilinen + sınırda) örneklerde doğruluk: **%{kalan_dogruluk * 100:.2f}**",
        f"- Bilinen örneklerde sınırda uyarısı oranı: **%{bilinen_sinirda_orani * 100:.2f}**",
        "",
        "## Not",
        "",
        "Eğitim setine Mendeley gerçek dünya görselleri (taze/çürük elma, muz,",
        "portakal) eklendiği için bilinen bölgenin mesafe dağılımı önceki modele",
        "göre değişmiştir; eşikler bu yeni dağılıma göre taranmıştır. Üç kademeli",
        "karar yapısı korunmuştur.",
    ]
    RAPOR_DOSYA.write_text("\n".join(satirlar), encoding="utf-8")

    print(f"\nRapor: {RAPOR_DOSYA}")
    if mevcut:
        print(
            f"Mevcut esikler: yanlis bilinmeyen %{mevcut['yanlis_bilinmeyen'] * 100:.2f}, "
            f"sinirda %{mevcut['sinirda'] * 100:.2f}, "
            f"dis: {mevcut['kesin_yakalanan']}/7 kesin + {mevcut['uyarili']}/7 sinirda, "
            f"ood_eval yanlis bilinmeyen %{mevcut['eval_yanlis_bilinmeyen'] * 100:.1f}, "
            f"uygun={mevcut['uygun']}"
        )
    print(f"Ham dogruluk: %{ham_dogruluk * 100:.2f} | OOD sonrasi: %{kalan_dogruluk * 100:.2f}")
    uygunlar = [r for r in izgara if r["uygun"]]
    print(f"Uygun aday sayisi: {len(uygunlar)}")
    for r in uygunlar[:10]:
        print(
            f"  T_dusuk={r['t_dusuk']:.3f} T_yuksek={r['t_yuksek']:.3f} "
            f"yanlis_bilinmeyen=%{r['yanlis_bilinmeyen'] * 100:.2f} "
            f"dis={r['kesin_yakalanan']}+{r['uyarili']}/7 "
            f"eval=%{r['eval_yanlis_bilinmeyen'] * 100:.1f}"
        )


if __name__ == "__main__":
    main()
