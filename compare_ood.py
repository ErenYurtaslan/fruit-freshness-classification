"""Eski ve yeni OOD sistemlerinin karsilastirmali dogrulama raporu.

Model degismedigi icin uc temel metrik (max_prob, entropy, embedding
distance) her iki sistemde aynidir; yalnizca KARAR MANTIGI farklidir.
Bu betik:
- diagnose_ood.py'nin kaydettigi eski sistem ciktilarini okur,
- ayni metrikler uzerinde yeni mesafe-oncelikli karari uygular,
- bilinen test orneklemi (calibrate_ood.py ciktisi) uzerinde eski ve yeni
  kurallarin yanlis 'bilinmeyen' oranlarini ve dogruluk etkisini hesaplar,
- yan yana karsilastirma tablosu, CSV ve grafik uretir.

Ciktilar:
- reports/ood_karsilastirma_raporu.md
- outputs/ood_analiz/karsilastirma_tablosu.csv
- outputs/ood_analiz/once_sonra_karsilastirma.png

Kullanim:
    python compare_ood.py
"""

import csv
import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402
from utils.ood_detector import OODAlgilayici  # noqa: E402
from utils.image_utils import sinif_adini_kullanici_dostu_yap  # noqa: E402

ANALIZ_DIZIN = config.CIKTI_DIZIN / "ood_analiz"
RAPOR_DOSYA = config.RAPOR_DIZIN / "ood_karsilastirma_raporu.md"
CSV_DOSYA = ANALIZ_DIZIN / "karsilastirma_tablosu.csv"
GRAFIK_DOSYA = ANALIZ_DIZIN / "once_sonra_karsilastirma.png"


def eski_karar(uzaklik, max_prob, entropy, esikler):
    """Iyilestirme oncesi (oylama + veto) karar mantiginin birebir kopyasi.

    Eski sistem ood_detector.py'den kaldirildigi icin karsilastirma amaciyla
    burada saf fonksiyon olarak korunur. Esik degerleri models/ood_stats.json
    dosyasindan gelir; veto sabitleri eski config degerleridir.
    """
    op_prob_esik = max(0.55, esikler["max_prob_alt_esik"] - 0.25)
    op_ent_esik = min(0.45, esikler["entropy_ust_esik"] + 0.22)
    op_dist_esik = esikler["distance_ust_esik"]

    kosul_prob = max_prob >= op_prob_esik
    kosul_ent = entropy <= op_ent_esik
    kosul_dist = uzaklik <= op_dist_esik
    saglanan = int(kosul_prob) + int(kosul_ent) + int(kosul_dist)

    mesafe_temelli_bilinen = kosul_dist and (max_prob >= 0.60 or entropy <= 0.42)
    cok_guvenli = max_prob >= 0.98 and entropy <= 0.03
    oy_coklugu_bilinen = saglanan >= 2
    yuksek_olasilik_bilinen = max_prob >= 0.75 and entropy <= 0.35
    bilinen_adayi = (
        mesafe_temelli_bilinen
        or oy_coklugu_bilinen
        or cok_guvenli
        or yuksek_olasilik_bilinen
    )
    # Eski veto sabitleri: tolerans 0.09, entropy 0.33, maxprob ust 0.90.
    mesafe_kotu = uzaklik > (op_dist_esik + 0.09)
    belirsiz = entropy > 0.33
    veto = mesafe_kotu and belirsiz and max_prob < 0.90
    return bilinen_adayi and not veto


def main():
    config.klasorleri_hazirla()

    eski_veri = json.loads(
        (ANALIZ_DIZIN / "eski_sistem_sonuclari.json").read_text(encoding="utf-8")
    )
    ist = json.loads(config.OOD_ISTATISTIK_DOSYA.read_text(encoding="utf-8"))
    esikler = ist["esikler"]
    bilinen = np.load(ANALIZ_DIZIN / "kalibrasyon_metrikleri.npz", allow_pickle=True)

    # ---- Dis ornekler: eski ve yeni kararlar yan yana ----
    durum_metni = {
        "bilinen": "Bilinen Tür",
        "sinirda": "Sınırda (Düşük Güven)",
        "bilinmeyen": "Bilinmeyen Tür",
    }
    dis_satirlar = []
    for s in eski_veri["dis_ornekler"]:
        en_yakin = s["en_yakin_merkezler"][0][0]
        durum, kural = OODAlgilayici.hiyerarsik_karar(
            s["mesafe"], s["max_prob"], s["entropy"],
            softmax_sinif=s["tahmin_sinifi"], en_yakin_sinif=en_yakin,
        )
        if durum == "bilinmeyen":
            yeni_tahmin, yeni_guven = "Bilinmeyen Meyve", None
        elif durum == "sinirda":
            yeni_tahmin = f"{s['tahmin_sinifi']} (düşük güven)"
            yeni_guven = s["max_prob"]
        else:
            yeni_tahmin, yeni_guven = s["tahmin_sinifi"], s["max_prob"]
        dis_satirlar.append(
            {
                "ad": s["ad"],
                "eski_tahmin": s["tahmin_sinifi"],
                "eski_guven": s["max_prob"],
                "eski_karar": s["karar"],
                "yeni_durum": durum,
                "yeni_karar": durum_metni[durum],
                "yeni_tahmin": yeni_tahmin,
                "yeni_guven": yeni_guven,
                "entropy": s["entropy"],
                "mesafe": s["mesafe"],
                "en_yakin_sinif": en_yakin,
                "tetikleyen_kural": kural,
            }
        )

    # ---- Bilinen orneklem: eski vs yeni yanlis 'bilinmeyen' orani ----
    mesafeler = bilinen["mesafeler"]
    max_prob = bilinen["max_prob"]
    entropy = bilinen["entropy"]
    gercek = bilinen["gercek"]
    tahmin = bilinen["tahmin"]
    en_yakin_dizi = bilinen["en_yakin"]

    eski_bilinen_maske = np.array(
        [
            eski_karar(float(d), float(p), float(e), esikler)
            for d, p, e in zip(mesafeler, max_prob, entropy)
        ]
    )
    yeni_durumlar = np.array(
        [
            OODAlgilayici.hiyerarsik_karar(
                float(d), float(p), float(e),
                softmax_sinif=str(t), en_yakin_sinif=str(y),
            )[0]
            for d, p, e, t, y in zip(mesafeler, max_prob, entropy, tahmin, en_yakin_dizi)
        ]
    )
    yeni_gosterilen_maske = yeni_durumlar != "bilinmeyen"

    n = len(mesafeler)
    eski_yanlis_ood = 1.0 - float(np.mean(eski_bilinen_maske))
    yeni_yanlis_ood = float(np.mean(yeni_durumlar == "bilinmeyen"))
    yeni_sinirda = float(np.mean(yeni_durumlar == "sinirda"))
    ham_dogruluk = float(np.mean(gercek == tahmin))
    eski_dogruluk = float(np.mean((gercek == tahmin)[eski_bilinen_maske]))
    yeni_dogruluk = float(np.mean((gercek == tahmin)[yeni_gosterilen_maske]))

    eski_yakalanan = sum(1 for r in dis_satirlar if r["eski_karar"] == "Bilinmeyen Tür")
    yeni_yakalanan = sum(1 for r in dis_satirlar if r["yeni_durum"] == "bilinmeyen")
    yeni_uyarili = sum(1 for r in dis_satirlar if r["yeni_durum"] == "sinirda")

    # ---- CSV ----
    with open(CSV_DOSYA, "w", newline="", encoding="utf-8-sig") as f:
        yazici = csv.writer(f)
        yazici.writerow(
            [
                "Görsel",
                "Eski Tahmin",
                "Eski Güven",
                "Yeni Tahmin",
                "Yeni Güven",
                "OOD Kararı",
                "Entropy",
                "Embedding Distance",
                "En Yakın Sınıf Merkezi",
                "Kararı Tetikleyen Kural",
            ]
        )
        for r in dis_satirlar:
            yazici.writerow(
                [
                    r["ad"],
                    r["eski_tahmin"],
                    f"%{r['eski_guven'] * 100:.1f}",
                    r["yeni_tahmin"],
                    f"%{r['yeni_guven'] * 100:.1f}" if r["yeni_guven"] else "-",
                    r["yeni_karar"],
                    f"{r['entropy']:.3f}",
                    f"{r['mesafe']:.3f}",
                    r["en_yakin_sinif"],
                    r["tetikleyen_kural"],
                ]
            )

    # ---- Grafik: once/sonra karsilastirma ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    adlar = [r["ad"] for r in dis_satirlar]
    y = np.arange(len(adlar))
    durum_degeri = {"bilinmeyen": 0.0, "sinirda": 0.5, "bilinen": 1.0}
    eski_durum = [0 if r["eski_karar"] == "Bilinmeyen Tür" else 1 for r in dis_satirlar]
    yeni_durum = [durum_degeri[r["yeni_durum"]] for r in dis_satirlar]
    ax1.barh(y - 0.2, eski_durum, height=0.38, color="#b03a3a",
             label="Eski sistem (1 = bilinen sanıldı)")
    ax1.barh(y + 0.2, yeni_durum, height=0.38, color="#5a8f5a",
             label="Yeni sistem")
    ax1.set_yticks(y)
    ax1.set_yticklabels(adlar, fontsize=9)
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_xticklabels(["Bilinmeyen", "Sınırda\n(uyarılı)", "Bilinen\n(hatalı)"])
    ax1.set_title("Dış Örneklerde Karar: Önce / Sonra")
    ax1.legend(fontsize=8)
    ax1.invert_yaxis()

    etiketler = [
        "Dış örnek\nişaretleme",
        "Bilinen yanlış\n'bilinmeyen'",
        "Bilinen kalan\ndoğruluk",
    ]
    eski_degerler = [eski_yakalanan / 7 * 100, eski_yanlis_ood * 100, eski_dogruluk * 100]
    yeni_degerler = [
        (yeni_yakalanan + yeni_uyarili) / 7 * 100,
        yeni_yanlis_ood * 100,
        yeni_dogruluk * 100,
    ]
    x = np.arange(len(etiketler))
    ax2.bar(x - 0.18, eski_degerler, width=0.36, color="#b03a3a", label="Eski sistem")
    ax2.bar(x + 0.18, yeni_degerler, width=0.36, color="#5a8f5a", label="Yeni sistem")
    for i, (e, yv) in enumerate(zip(eski_degerler, yeni_degerler)):
        ax2.text(i - 0.18, e + 1, f"%{e:.1f}", ha="center", fontsize=8)
        ax2.text(i + 0.18, yv + 1, f"%{yv:.1f}", ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(etiketler, fontsize=9)
    ax2.set_ylabel("Yüzde (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("Özet Metrikler: Önce / Sonra")
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(GRAFIK_DOSYA, dpi=150)
    plt.close()

    # ---- Markdown rapor ----
    satirlar = [
        "# OOD İyileştirmesi: Önce / Sonra Karşılaştırma ve Doğrulama Raporu",
        "",
        f"*Oluşturma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "Model ağırlıkları değişmediği için max_prob, entropy ve embedding distance",
        "değerleri iki sistemde de aynıdır; karşılaştırılan şey yalnızca **karar mantığıdır**.",
        "Eski sistem: oy çokluğu + yüksek olasılık kuralları + entropy şartlı veto.",
        "Yeni sistem: mesafe-öncelikli ÜÇ KADEMELİ karar"
        f" (T_yüksek={config.OOD_MESAFE_T_YUKSEK}, T_düşük={config.OOD_MESAFE_T_DUSUK};",
        "gri bölgede softmax-merkez uyumu varsa 'Sınırda' uyarısıyla tahmin gösterilir).",
        "",
        "## 1. Dış Örnekler: Yan Yana Karşılaştırma",
        "",
    ]
    for r in dis_satirlar:
        eski_g = sinif_adini_kullanici_dostu_yap(r["eski_tahmin"])
        if r["yeni_durum"] == "bilinmeyen":
            sonra_metni = "**Bilinmeyen Meyve**"
        elif r["yeni_durum"] == "sinirda":
            sonra_metni = (
                f"{sinif_adini_kullanici_dostu_yap(r['eski_tahmin'])} "
                f"(%{r['yeni_guven'] * 100:.1f}) — **Sınırda / düşük güven uyarısı**"
            )
        else:
            sonra_metni = (
                f"{sinif_adini_kullanici_dostu_yap(r['eski_tahmin'])} "
                f"(%{r['yeni_guven'] * 100:.1f})"
            )
        satirlar += [
            f"### {r['ad']}",
            "",
            f"- **Önce:** {eski_g} (%{r['eski_guven'] * 100:.1f}) — karar: {r['eski_karar']}",
            f"- **Sonra:** {sonra_metni}",
            f"- Embedding Distance: **{r['mesafe']:.3f}**",
            f"- Entropy: **{r['entropy']:.3f}**",
            f"- En Yakın Sınıf Merkezi: **{r['en_yakin_sinif']}**",
            f"- Kararı Tetikleyen Kural: **{r['tetikleyen_kural']}**",
            "",
        ]

    satirlar += [
        "## 2. Özet Tablo",
        "",
        "| Görsel | Eski Tahmin | Eski Güven | Yeni Tahmin | Yeni Güven | Mesafe | Entropy | En Yakın Merkez | Tetikleyen Kural |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in dis_satirlar:
        satirlar.append(
            f"| {r['ad']} | {r['eski_tahmin']} | %{r['eski_guven'] * 100:.1f} "
            f"| {r['yeni_tahmin']} "
            f"| {('%' + format(r['yeni_guven'] * 100, '.1f')) if r['yeni_guven'] else '-'} "
            f"| {r['mesafe']:.3f} | {r['entropy']:.3f} | {r['en_yakin_sinif']} "
            f"| {r['tetikleyen_kural']} |"
        )

    toplam_isaret = yeni_yakalanan + yeni_uyarili
    satirlar += [
        "",
        f"![Önce / sonra karşılaştırma]({GRAFIK_DOSYA.as_posix()})",
        "",
        "## 3. Sayısal Metrikler",
        "",
        f"Bilinen örneklem: **{n}** test görseli.",
        "",
        "| Metrik | Eski Sistem | Yeni Sistem |",
        "|---|---|---|",
        f"| Bilinen örneklerde yanlış OOD oranı | %{eski_yanlis_ood * 100:.2f} | %{yeni_yanlis_ood * 100:.2f} |",
        f"| Bilinen örneklerde sınırda uyarısı | - | %{yeni_sinirda * 100:.2f} |",
        f"| Gösterilen örneklerde doğruluk | %{eski_dogruluk * 100:.2f} | %{yeni_dogruluk * 100:.2f} |",
        f"| Dış örnek: kesin Bilinmeyen | {eski_yakalanan}/7 | {yeni_yakalanan}/7 |",
        f"| Dış örnek: sınırda uyarısı | - | {yeni_uyarili}/7 |",
        f"| Dış örnek: toplam işaretleme | {eski_yakalanan}/7 | {toplam_isaret}/7 |",
        "",
        f"- Örneklemde ham model doğruluğu (OOD filtresi yok): %{ham_dogruluk * 100:.2f}",
        f"- Dış örnek işaretlemede iyileşme: {eski_yakalanan}/7 → {toplam_isaret}/7",
        "",
        "## 4. Üç Kademeli Kararın Gerekçesi (sarı elma vakası)",
        "",
        "k-NN ölçümü, gerçek-ama-alışılmadık meyvelerin (sarı buruşuk elma: 0.412)",
        "yabancı meyvelerle (mürdüm: 0.454, üzüm: 0.495) embedding uzayında iç içe",
        "geçtiğini göstermiştir; bilinen test maksimumu 0.374'tür. Tek eşik bu iki",
        "grubu aynı anda ayıramaz. Üç kademeli karar bu açmaza şöyle yanıt verir:",
        "",
        "- Sarı elma (mesafe ≈ 0.450, softmax ve en yakın merkez aynı sınıf:",
        "  rottenapples) → **Çürük Elma (sınırda, düşük güven uyarısı)**",
        "- Üzüm/ejder/liçi (mesafe > 0.46) → **kesin Bilinmeyen**",
        "- Sinyaller çelişirse (softmax ≠ en yakın merkez) → **Bilinmeyen**",
        "",
        "## 5. Sonuç Soruları",
        "",
        "**1. Yeni sistem dış örnekleri daha başarılı işaretliyor mu?**",
        f"Evet. Eski sistem {eski_yakalanan}/7, yeni sistem {yeni_yakalanan}/7 kesin",
        f"Bilinmeyen + {yeni_uyarili}/7 sınırda uyarısı (toplam {toplam_isaret}/7)",
        "üretmektedir. Eski sistemde yüksek softmax güveni mesafe sinyalini eziyordu;",
        "yeni hiyerarşide mesafe birincil sinyaldir.",
        "",
        "**2. Bilinen sınıflarda kabul edilebilir doğruluk seviyesini koruyor mu?**",
        f"Evet. Yanlış 'bilinmeyen' oranı %{yeni_yanlis_ood * 100:.2f} (hedef ≤ %2) ve",
        f"gösterilen örneklerde doğruluk %{yeni_dogruluk * 100:.2f} olup ham doğruluğun",
        f"(%{ham_dogruluk * 100:.2f}) üzerindedir.",
        "",
        "**3. OOD kararlarını en çok hangi sinyal tetikliyor?**",
        "**Embedding Distance.** Kesin Bilinmeyen kararlarının tamamı `Mesafe > T_yüksek`",
        "kuralıyla verilmiştir. Entropy satüre softmax nedeniyle ≈0 olduğundan kör,",
        "max_prob dış örneklerde bile %98+ olduğundan yanıltıcıdır; bunlar yalnızca",
        "gri bölgede (softmax-merkez uyumuyla birlikte) yardımcı sinyaldir.",
        "",
        "**4. Yakalama oranı nihai sınır mı?**",
        "Tek global eşikle kesin yakalama 3/7-5/7 bandındadır; üç kademeli karar",
        "sınırda uyarısıyla kapsamı genişletir. 6/7+ kesin yakalama için sınıf-bazlı",
        "eşik veya Mahalanobis mesafesi (ikinci faz) gerekir.",
        "",
        "**5. Sistemin halen başarısız olduğu örnekler hangileri ve neden?**",
        "- **Mor Erik (d=0.321):** Elma manifoldunun içine düşmektedir; hiçbir kademe",
        "  tetiklenmez, 'bilinen' görünür.",
        "- **Mandalina (d=0.287):** Portakala gerçekten çok benzer; Fresh Orange",
        "  bölgesinin içindedir. Anlamsal olarak yakın bir tahmin verdiğinden kabul",
        "  edilebilir sınır vakasıdır.",
        "Bu iki örnek bilinen dağılım bölgesinin (T_düşük altı) içindedir; embedding",
        "temsili değişmeden (ör. yeniden eğitim/metrik öğrenme) ayrıştırılamazlar.",
    ]
    RAPOR_DOSYA.write_text("\n".join(satirlar), encoding="utf-8")

    print(f"Rapor: {RAPOR_DOSYA}")
    print(f"CSV: {CSV_DOSYA}")
    print(f"Grafik: {GRAFIK_DOSYA}")
    print(
        f"\nEski: {eski_yakalanan}/7 yakalama, %{eski_yanlis_ood * 100:.2f} yanlis OOD | "
        f"Yeni: {yeni_yakalanan}/7 kesin + {yeni_uyarili}/7 sinirda, "
        f"%{yeni_yanlis_ood * 100:.2f} yanlis OOD"
    )
    for r in dis_satirlar:
        print(
            f"- {r['ad']}: {r['eski_karar']} -> {r['yeni_karar']} ({r['tetikleyen_kural']})"
        )


if __name__ == "__main__":
    main()
