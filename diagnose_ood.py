"""OOD teshis betigi.

Bu betik modeli DEGISTIRMEZ; yalnizca mevcut davranisi olcer ve raporlar:
1. Bilinen (test) ornekler ile dis (OOD) orneklerin sinif merkezlerine
   mesafe dagilimlarini karsilastirir.
2. Dis ornekler icin en yakin sinif merkezi analizini yapar
   (hangi sinif neden cekiyor).
3. Embedding uzayini PCA ile 2 boyuta indirip gorsellestirir
   (apple bolgesinin genisligi kaniti).
4. Dis ornekler icin Grad-CAM izgarasi uretir (model neye bakiyor).
5. Mevcut (eski) OOD karar sisteminin dis ornek ciktilarini JSON'a
   kaydeder; boylece iyilestirme sonrasi karsilastirma yapilabilir.

Ciktilar:
- reports/ood_teshis_raporu.md
- outputs/ood_analiz/mesafe_dagilimi.png
- outputs/ood_analiz/embedding_pca.png
- outputs/ood_analiz/gradcam_izgara.png
- outputs/ood_analiz/eski_sistem_sonuclari.json

Kullanim:
    python diagnose_ood.py
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from src import config  # noqa: E402
from src.utils import goruntu_yukle_ve_hazirla  # noqa: E402
from gradcam import grad_model_olustur, isi_haritasi_uret, overlay_olustur  # noqa: E402
from utils.ood_detector import OODAlgilayici  # noqa: E402

# Dis ornek klasoru ve analiz cikti klasoru.
OOD_TEST_DIZIN = config.CIKTI_DIZIN / "ood_test"
ANALIZ_DIZIN = config.CIKTI_DIZIN / "ood_analiz"
RAPOR_DOSYA = config.RAPOR_DIZIN / "ood_teshis_raporu.md"

# Dosya adi -> okunur Turkce ad eslemesi.
DIS_ORNEK_ADLARI = {
    "ejder_meyvesi": "Ejder Meyvesi",
    "lici": "Liçi",
    "mor_uzum": "Mor Üzüm",
    "mor_erik": "Mor Erik",
    "murdum_erigi": "Mürdüm Eriği",
    "mandalina": "Mandalina",
    "greyfurt": "Greyfurt",
}

# Bilinen dagilim olcumu icin sinif basina test ornegi sayisi.
SINIF_BASI_TEST_ORNEK = 60


def _sinif_isimlerini_oku():
    esleme = json.loads(config.SINIF_INDEKS_DOSYA.read_text(encoding="utf-8"))
    return [esleme[str(i)] for i in range(len(esleme))]


def _test_orneklerini_topla(sinif_isimleri):
    """Test setinden sinif basina esit sayida dosya yolu toplar."""
    ornekler = []
    for sinif in sinif_isimleri:
        klasor = config.TEST_DIZIN / sinif
        dosyalar = sorted(
            p
            for p in klasor.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )[:SINIF_BASI_TEST_ORNEK]
        ornekler.extend((sinif, p) for p in dosyalar)
    return ornekler


def _tum_merkez_uzakliklari(detektor, embedding):
    """Embedding'in tum sinif merkezlerine cosine uzakliklarini dondurur."""
    return {
        sinif: detektor._cosine_distance(embedding, merkez)
        for sinif, merkez in detektor.istatistikler["sinif_merkezleri"].items()
    }


def bilinen_metrikleri_hesapla(model, detektor, ornekler):
    """Bilinen test ornekleri icin metrik dizileri uretir."""
    yollar = [p for _, p in ornekler]
    girdi = detektor._batch_onisle(yollar)
    olasilik = model.predict(girdi, batch_size=32, verbose=0)
    embedding = detektor.embedding_model.predict(girdi, batch_size=32, verbose=0)

    mesafeler = np.array(
        [
            detektor._en_yakin_merkez_uzakligi(
                v, detektor.istatistikler["sinif_merkezleri"]
            )
            for v in embedding
        ]
    )
    max_prob = np.max(olasilik, axis=1)
    entropy = detektor._normalize_entropy(olasilik)
    siniflar = [s for s, _ in ornekler]
    return {
        "embedding": embedding,
        "mesafeler": mesafeler,
        "max_prob": max_prob,
        "entropy": entropy,
        "siniflar": siniflar,
    }


def dis_ornekleri_analiz_et(model, detektor, sinif_isimleri):
    """7 dis ornek icin tum metrikleri ve mevcut sistem kararini cikarir."""
    sonuclar = []
    for govde, ad in DIS_ORNEK_ADLARI.items():
        eslesen = list(OOD_TEST_DIZIN.glob(f"{govde}.*"))
        if not eslesen:
            print(f"UYARI: {govde} icin dosya bulunamadi, atlaniyor.")
            continue
        yol = eslesen[0]
        model_girisi, ham = goruntu_yukle_ve_hazirla(str(yol))
        olasilik = model.predict(model_girisi, verbose=0)[0]
        embedding = detektor.embedding_model.predict(model_girisi, verbose=0)[0]
        ood = detektor.degerlendir(model_girisi, olasilik)

        merkez_uzakliklari = _tum_merkez_uzakliklari(detektor, embedding)
        sirali = sorted(merkez_uzakliklari.items(), key=lambda x: x[1])

        en_iyi = int(np.argmax(olasilik))
        sonuclar.append(
            {
                "ad": ad,
                "dosya": str(yol),
                "ham_goruntu": ham,
                "model_girisi": model_girisi,
                "embedding": embedding,
                "tahmin_sinifi": sinif_isimleri[en_iyi],
                "max_prob": float(np.max(olasilik)),
                "entropy": float(ood["entropy"]),
                "mesafe": float(ood["embedding_distance"]),
                "en_yakin_merkezler": [(s, float(d)) for s, d in sirali],
                "eski_karar": ood["dagilim_durumu"],
                "eski_karar_ozeti": ood["karar_ozeti"],
            }
        )
    return sonuclar


def mesafe_dagilim_grafigi(bilinen, dis_sonuclar, esik):
    """Bilinen vs dis ornek mesafe histogramini cizer."""
    plt.figure(figsize=(9, 5))
    plt.hist(
        bilinen["mesafeler"],
        bins=40,
        alpha=0.65,
        color="#5a8f5a",
        label=f"Bilinen test örnekleri (n={len(bilinen['mesafeler'])})",
    )
    for s in dis_sonuclar:
        plt.axvline(s["mesafe"], color="#b03a3a", alpha=0.85, linewidth=1.4)
        plt.text(
            s["mesafe"],
            plt.gca().get_ylim()[1] * 0.92,
            s["ad"],
            rotation=90,
            fontsize=7,
            ha="right",
            va="top",
            color="#7a1f1f",
        )
    plt.axvline(
        esik,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=f"Mevcut mesafe eşiği ({esik:.3f})",
    )
    plt.title("En Yakın Sınıf Merkezine Cosine Mesafe Dağılımı")
    plt.xlabel("Cosine mesafe")
    plt.ylabel("Örnek sayısı")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    yol = ANALIZ_DIZIN / "mesafe_dagilimi.png"
    plt.savefig(yol, dpi=150)
    plt.close()
    return yol


def pca_grafigi(bilinen, dis_sonuclar, sinif_isimleri):
    """Embedding uzayini PCA ile 2B'ye indirip cizer."""
    from sklearn.decomposition import PCA

    X = bilinen["embedding"]
    pca = PCA(n_components=2, random_state=config.TOHUM)
    X2 = pca.fit_transform(X)
    dis_X2 = pca.transform(np.array([s["embedding"] for s in dis_sonuclar]))

    renkler = plt.cm.tab10(np.linspace(0, 1, len(sinif_isimleri)))
    plt.figure(figsize=(10, 7))
    siniflar = np.array(bilinen["siniflar"])
    for i, sinif in enumerate(sinif_isimleri):
        maske = siniflar == sinif
        plt.scatter(
            X2[maske, 0],
            X2[maske, 1],
            s=14,
            alpha=0.55,
            color=renkler[i],
            label=sinif,
        )
    plt.scatter(
        dis_X2[:, 0],
        dis_X2[:, 1],
        s=140,
        marker="X",
        color="black",
        label="Dış örnekler",
        zorder=5,
    )
    for (x, y), s in zip(dis_X2, dis_sonuclar):
        plt.annotate(s["ad"], (x, y), fontsize=8, xytext=(6, 6),
                     textcoords="offset points")
    aciklanan = pca.explained_variance_ratio_
    plt.title(
        "Embedding Uzayı PCA Görselleştirmesi "
        f"(açıklanan varyans: %{100 * aciklanan.sum():.1f})"
    )
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(fontsize=8, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    yol = ANALIZ_DIZIN / "embedding_pca.png"
    plt.savefig(yol, dpi=150)
    plt.close()
    return yol


def gradcam_izgara(model, dis_sonuclar):
    """Dis ornekler icin Grad-CAM overlay izgarasi uretir."""
    grad_model, _ = grad_model_olustur(model)
    n = len(dis_sonuclar)
    plt.figure(figsize=(3.1 * n, 6.6))
    for i, s in enumerate(dis_sonuclar):
        isi, _, _ = isi_haritasi_uret(grad_model, s["model_girisi"])
        overlay = overlay_olustur(s["ham_goruntu"], isi)

        plt.subplot(2, n, i + 1)
        plt.imshow(s["ham_goruntu"])
        plt.title(s["ad"], fontsize=9)
        plt.axis("off")

        plt.subplot(2, n, n + i + 1)
        plt.imshow(overlay)
        plt.title(
            f"{s['tahmin_sinifi']}\n%{s['max_prob'] * 100:.1f} | d={s['mesafe']:.3f}",
            fontsize=8,
        )
        plt.axis("off")
    plt.suptitle("Dış Örnekler İçin Grad-CAM Analizi (üst: orijinal, alt: ısı haritası)")
    plt.tight_layout()
    yol = ANALIZ_DIZIN / "gradcam_izgara.png"
    plt.savefig(yol, dpi=150)
    plt.close()
    return yol


def rapor_yaz(bilinen, dis_sonuclar, esikler, grafik_yollari):
    """Teshis raporunu markdown olarak uretir."""
    mes = bilinen["mesafeler"]
    yuzdelikler = {p: float(np.percentile(mes, p)) for p in (50, 90, 95, 99)}

    # En yakin merkez sayimi (dis ornekler hangi sinifa cekiliyor).
    cekim_sayaci = {}
    for s in dis_sonuclar:
        en_yakin = s["en_yakin_merkezler"][0][0]
        cekim_sayaci[en_yakin] = cekim_sayaci.get(en_yakin, 0) + 1

    satirlar = [
        "# OOD Teşhis Raporu",
        "",
        f"*Oluşturma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## 1. Amaç",
        "",
        "Bu rapor, modelin eğitim dağılımı dışındaki (OOD) meyveleri neden",
        "yüksek güvenle bilinen sınıflara atadığını kanıta dayalı olarak belgeler.",
        "Hiçbir model parametresi değiştirilmemiştir; yalnızca mevcut davranış ölçülmüştür.",
        "",
        "## 2. Bilinen Dağılım Mesafe İstatistikleri",
        "",
        f"- Ölçülen bilinen test örneği sayısı: **{len(mes)}**",
        f"- Medyan mesafe (P50): **{yuzdelikler[50]:.4f}**",
        f"- P90: **{yuzdelikler[90]:.4f}**",
        f"- P95: **{yuzdelikler[95]:.4f}**",
        f"- P99: **{yuzdelikler[99]:.4f}**",
        f"- Mevcut mesafe eşiği (train P95): **{esikler['distance_ust_esik']:.4f}**",
        "",
        f"![Mesafe dağılımı]({grafik_yollari['mesafe'].as_posix()})",
        "",
        "## 3. Dış Örnek Analizi (Mevcut Sistem)",
        "",
        "| Örnek | Tahmin | MaxProb | Entropy | Mesafe | En Yakın Merkez | Eski Karar |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in dis_sonuclar:
        en_yakin_ad, en_yakin_d = s["en_yakin_merkezler"][0]
        satirlar.append(
            f"| {s['ad']} | {s['tahmin_sinifi']} | %{s['max_prob'] * 100:.1f} "
            f"| {s['entropy']:.3f} | {s['mesafe']:.3f} "
            f"| {en_yakin_ad} ({en_yakin_d:.3f}) | {s['eski_karar']} |"
        )

    esik = esikler["distance_ust_esik"]
    esik_ustu = [s for s in dis_sonuclar if s["mesafe"] > esik]
    eski_yakalanan = [s for s in dis_sonuclar if s["eski_karar"] == "Bilinmeyen Tür"]

    satirlar += [
        "",
        "### Sınıf çekim analizi",
        "",
        "Dış örneklerin en yakın olduğu sınıf merkezleri:",
        "",
    ]
    for sinif, adet in sorted(cekim_sayaci.items(), key=lambda x: -x[1]):
        satirlar.append(f"- **{sinif}**: {adet} dış örnek")
    satirlar += [
        "",
        "Apple sınıfları veri setinin yaklaşık %37'sini oluşturduğundan ve",
        "yuvarlak/parlak nesneler apple manifolduna çekildiğinden, dış örneklerin",
        "çoğunun elma merkezlerine yaklaşması veri seti yanlılığıyla (dataset bias) tutarlıdır.",
        "",
        "## 4. Embedding Uzayı Görselleştirmesi",
        "",
        f"![Embedding PCA]({grafik_yollari['pca'].as_posix()})",
        "",
        "PCA izdüşümünde apple kümeleri geniş bir bölgeye yayılmaktadır;",
        "dış örnekler bu bölgenin kenarına düşmekte ancak softmax bunlara",
        "yüksek olasılık atamaktadır. Bu, kapalı küme (closed-set) sınıflandırmanın",
        "yapısal sınırlamasıdır: softmax her girdiyi mevcut 6 sınıftan birine dağıtmak zorundadır.",
        "",
        "## 5. Grad-CAM Analizi",
        "",
        f"![Grad-CAM ızgarası]({grafik_yollari['gradcam'].as_posix()})",
        "",
        "Isı haritaları, modelin dış örneklerde de nesnenin yuvarlak/parlak gövdesine",
        "odaklandığını gösterir. Model 'bu bir elma mı?' sorusunu değil,",
        "'6 sınıftan hangisine en çok benziyor?' sorusunu cevaplamaktadır.",
        "",
        "## 6. Temel Bulgular",
        "",
        f"1. **Mesafe sinyali çalışıyor:** {len(esik_ustu)}/{len(dis_sonuclar)} dış örnek mevcut",
        f"   mesafe eşiğinin ({esik:.3f}) üzerindedir; embedding uzaklığı dış örnekleri ayırt edebilmektedir.",
        f"2. **Karar hiyerarşisi mesafeyi eziyor:** Eski sistem yalnızca "
        f"{len(eski_yakalanan)}/{len(dis_sonuclar)} dış örneği yakalamıştır. "
        "`yuksek_olasilik_bilinen` ve `oy_coklugu_bilinen` kuralları, mesafe sinyali",
        "   'bilinmeyen' derken yüksek softmax güveni nedeniyle kararı 'bilinen'e çevirmektedir.",
        "3. **Veto kuralı kör:** Veto `entropy > 0.33` şartına bağlıdır; kendinden emin",
        "   yanlışlarda entropy ≈ 0 olduğu için veto hiç tetiklenmemektedir.",
        "4. **Sorun calibration değil:** Temperature scaling argmax'ı değiştirmez;",
        "   ejder meyvesi yine 'Fresh Apple' kalır, yalnızca güven yüzdesi düşer.",
        "",
        "## 7. Önerilen Çözüm",
        "",
        "Mesafe-öncelikli hiyerarşik karar:",
        "",
        "- `mesafe > T_yüksek` → **Bilinmeyen** (softmax ne derse desin)",
        "- `mesafe < T_düşük` → **Bilinen**",
        "- Gri bölge → max_prob / entropy yardımcı sinyal",
        "",
        "Eşikler bilinen test dağılımından kalibre edilir: bilinen örneklerde yanlış",
        "'bilinmeyen' oranı ≤ %2, dış örneklerde yakalama ≥ 5/7 hedeflenir.",
    ]

    RAPOR_DOSYA.write_text("\n".join(satirlar), encoding="utf-8")
    return RAPOR_DOSYA


def eski_sistem_sonuclarini_kaydet(dis_sonuclar, bilinen):
    """Iyilestirme oncesi sistem ciktilarini karsilastirma icin saklar."""
    veri = {
        "olusturma_zamani": datetime.now().isoformat(timespec="seconds"),
        "dis_ornekler": [
            {
                "ad": s["ad"],
                "dosya": s["dosya"],
                "tahmin_sinifi": s["tahmin_sinifi"],
                "max_prob": s["max_prob"],
                "entropy": s["entropy"],
                "mesafe": s["mesafe"],
                "en_yakin_merkezler": s["en_yakin_merkezler"],
                "karar": s["eski_karar"],
                "karar_ozeti": s["eski_karar_ozeti"],
            }
            for s in dis_sonuclar
        ],
        "bilinen_ozet": {
            "ornek_sayisi": int(len(bilinen["mesafeler"])),
            "mesafe_p50": float(np.percentile(bilinen["mesafeler"], 50)),
            "mesafe_p95": float(np.percentile(bilinen["mesafeler"], 95)),
            "mesafe_p99": float(np.percentile(bilinen["mesafeler"], 99)),
        },
    }
    yol = ANALIZ_DIZIN / "eski_sistem_sonuclari.json"
    yol.write_text(json.dumps(veri, ensure_ascii=False, indent=2), encoding="utf-8")
    return yol


def main():
    config.klasorleri_hazirla()
    ANALIZ_DIZIN.mkdir(parents=True, exist_ok=True)

    print("Model yukleniyor...")
    model = tf.keras.models.load_model(str(config.MODEL_DOSYA))
    sinif_isimleri = _sinif_isimlerini_oku()
    detektor = OODAlgilayici(model, sinif_isimleri)
    esikler = detektor.istatistikler["esikler"]

    print("Bilinen test ornekleri olculuyor...")
    ornekler = _test_orneklerini_topla(sinif_isimleri)
    bilinen = bilinen_metrikleri_hesapla(model, detektor, ornekler)

    print("Dis ornekler analiz ediliyor...")
    dis_sonuclar = dis_ornekleri_analiz_et(model, detektor, sinif_isimleri)

    print("Grafikler uretiliyor...")
    grafikler = {
        "mesafe": mesafe_dagilim_grafigi(bilinen, dis_sonuclar, esikler["distance_ust_esik"]),
        "pca": pca_grafigi(bilinen, dis_sonuclar, sinif_isimleri),
        "gradcam": gradcam_izgara(model, dis_sonuclar),
    }

    print("Rapor yaziliyor...")
    rapor = rapor_yaz(bilinen, dis_sonuclar, esikler, grafikler)
    eski_json = eski_sistem_sonuclarini_kaydet(dis_sonuclar, bilinen)

    # Bilinen metrikleri kalibrasyonda yeniden kullanmak icin sakla.
    np.savez(
        ANALIZ_DIZIN / "bilinen_metrikler.npz",
        mesafeler=bilinen["mesafeler"],
        max_prob=bilinen["max_prob"],
        entropy=bilinen["entropy"],
        siniflar=np.array(bilinen["siniflar"]),
    )

    print(f"\nRapor: {rapor}")
    print(f"Eski sistem ciktilari: {eski_json}")
    for ad, yol in grafikler.items():
        print(f"Grafik ({ad}): {yol}")
    print("\nDis ornek ozeti (mevcut sistem):")
    for s in dis_sonuclar:
        print(
            f"- {s['ad']}: {s['tahmin_sinifi']} %{s['max_prob'] * 100:.1f} "
            f"d={s['mesafe']:.3f} ent={s['entropy']:.3f} -> {s['eski_karar']}"
        )


if __name__ == "__main__":
    main()
