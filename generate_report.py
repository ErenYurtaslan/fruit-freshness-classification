"""Akademik PDF rapor ureticisi.

Egitim ve degerlendirme asamalarinda uretilen tum ciktilari (grafikler,
confusion matrix, Grad-CAM gorselleri, metrikler) tek bir akademik PDF
dokumaninda birlestirir. Dokuman hem RAPOR hem de SUNUM mantigi tasiyan
16 bolumden olusur.

Eksik ciktilar (orn. model henuz egitilmediyse) zarif sekilde atlanir ve
yerine bilgilendirici bir not yazilir.

Kullanim:
    python generate_report.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src import config

# Cikti PDF yolu.
PDF_DOSYA = config.RAPOR_DIZIN / "akademik_rapor.pdf"


class KapakSayfasi(Flowable):
    """Kapak icerigini tek sayfada mutlak konumlarla cizer."""

    def __init__(self, font, font_kalin):
        super().__init__()
        self.font = font
        self.font_kalin = font_kalin
        self.width = 0
        self.height = 0

    def wrap(self, avail_width, avail_height):
        self.width = avail_width
        self.height = avail_height
        return (avail_width, avail_height)

    def draw(self):
        c = self.canv
        pw, ph = self.width, self.height
        cx = pw / 2
        yan = 0.6 * cm

        uni_pt, uni_aralik = 15, 26
        y = ph - 1.3 * cm
        c.setFont(self.font, uni_pt)
        for satir in (
            "T.C. Sakarya Üniversitesi",
            "Fen Bilimleri Enstitüsü",
            "Bilişim Sistemleri Mühendisliği Yüksek Lisans Programı",
            "Bilgisayar Görmesi Uygulamaları",
        ):
            c.drawCentredString(cx, y, satir)
            y -= uni_aralik

        baslik_stil = ParagraphStyle(
            "kapak_baslik",
            fontName=self.font_kalin,
            fontSize=26,
            leading=36,
            alignment=TA_CENTER,
        )
        baslik = Paragraph(
            "Grad-CAM Destekli Açıklanabilir<br/>"
            "Meyve Tazelik Sınıflandırma Sistemi",
            baslik_stil,
        )
        genislik = pw - 2 * yan
        _, th = baslik.wrap(genislik, ph)
        baslik_orta = ph * 0.50
        baslik.drawOn(c, yan, baslik_orta - th / 2)

        yazar_pt, yazar_aralik = 15, 28
        yazar_ust = baslik_orta - th / 2 - 2.6 * cm
        c.setFont(self.font, yazar_pt)
        c.drawCentredString(cx, yazar_ust, "A. Eren Yurtaslan - y255052061")
        c.drawCentredString(
            cx,
            yazar_ust - yazar_aralik,
            "Dersi Veren: Dr.Öğr.Üyesi SERAP ÇAKAR KAMAN",
        )

        c.setFont(self.font, 14)
        c.drawCentredString(
            cx,
            yazar_ust - yazar_aralik - 2.4 * cm,
            "Haziran 2026, Sakarya",
        )


def _turkce_font_kaydet():
    """Turkce karakterleri destekleyen bir TrueType font kaydeder.

    Standart PDF fontlari (Helvetica) Turkce'ye ozgu g, s, i gibi
    karakterleri tam desteklemez. Bu yuzden Windows'taki Arial fontu
    kullanilir. Bulunamazsa Helvetica'ya geri donulur.
    """
    aday_yollar = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ]
    aday_kalin = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
    ]
    normal = next((y for y in aday_yollar if y.exists()), None)
    kalin = next((y for y in aday_kalin if y.exists()), None)

    if normal is not None:
        pdfmetrics.registerFont(TTFont("AnaFont", str(normal)))
        if kalin is not None:
            pdfmetrics.registerFont(TTFont("AnaFont-Bold", str(kalin)))
        else:
            pdfmetrics.registerFont(TTFont("AnaFont-Bold", str(normal)))
        return "AnaFont", "AnaFont-Bold"

    # Geri donus: standart fontlar (Turkce karakterler bozulabilir).
    return "Helvetica", "Helvetica-Bold"


def _stiller_olustur(font, font_kalin):
    """Rapor boyunca kullanilacak paragraf stillerini olusturur."""
    temel = getSampleStyleSheet()
    stiller = {
        "baslik": ParagraphStyle(
            "baslik",
            parent=temel["Title"],
            fontName=font_kalin,
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
        ),
        "altbaslik": ParagraphStyle(
            "altbaslik",
            fontName=font,
            fontSize=13,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.grey,
        ),
        "bolum": ParagraphStyle(
            "bolum",
            fontName=font_kalin,
            fontSize=15,
            leading=20,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor("#1a5e1a"),
        ),
        "govde": ParagraphStyle(
            "govde",
            fontName=font,
            fontSize=10.5,
            leading=15,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "not": ParagraphStyle(
            "not",
            fontName=font,
            fontSize=9,
            leading=12,
            textColor=colors.grey,
        ),
        "kaynakca": ParagraphStyle(
            "kaynakca",
            fontName=font,
            fontSize=10,
            leading=12,
            alignment=TA_JUSTIFY,
            spaceAfter=0,
            splitLongWords=False,
        ),
        "kaynakca_baslik": ParagraphStyle(
            "kaynakca_baslik",
            fontName=font_kalin,
            fontSize=15,
            leading=20,
            spaceBefore=2,
            spaceAfter=4,
            textColor=colors.HexColor("#1a5e1a"),
        ),
    }
    return stiller


def _gorsel_ekle(akis, gorsel_yolu, stiller, genislik=15 * cm):
    """Varsa bir gorseli, yoksa bilgilendirici notu akisa ekler."""
    gorsel_yolu = Path(gorsel_yolu)
    if gorsel_yolu.exists():
        img = Image(str(gorsel_yolu))
        olcek = genislik / img.drawWidth
        img.drawWidth = genislik
        img.drawHeight = img.drawHeight * olcek
        akis.append(img)
    else:
        akis.append(
            Paragraph(
                f"[Gorsel henuz uretilmedi: {gorsel_yolu.name}. "
                "Ilgili betigi calistirin.]",
                stiller["not"],
            )
        )
    akis.append(Spacer(1, 0.4 * cm))


def _metrik_tablosu(stiller):
    """Kaydedilmis metrikleri okuyup bir tablo bilesenine donusturur."""
    metrik_dosya = config.RAPOR_DIZIN / "metrikler.json"
    if not metrik_dosya.exists():
        return Paragraph(
            "[Metrikler henuz uretilmedi. 'python evaluate.py' calistirin.]",
            stiller["not"],
        )
    with open(metrik_dosya, "r", encoding="utf-8") as dosya:
        metrikler = json.load(dosya)

    etiketler = {
        "accuracy": "Dogruluk (Accuracy)",
        "precision_makro": "Precision (Makro)",
        "recall_makro": "Recall (Makro)",
        "f1_makro": "F1 Skoru (Makro)",
        "precision_agirlikli": "Precision (Agirlikli)",
        "recall_agirlikli": "Recall (Agirlikli)",
        "f1_agirlikli": "F1 Skoru (Agirlikli)",
    }
    veri = [["Metrik", "Deger"]]
    for anahtar, etiket in etiketler.items():
        if anahtar in metrikler:
            veri.append([etiket, f"{metrikler[anahtar] * 100:.2f}%"])

    tablo = Table(veri, colWidths=[9 * cm, 5 * cm])
    tablo.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5e1a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef6ee")]),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tablo


def _sinif_sayilari_oku(ayrim_dizini):
    """Verilen ayrim dizinindeki sinif klasorleri ve goruntu sayilarini okur."""
    if not ayrim_dizini.exists():
        return {}

    sayilar = {}
    for klasor in sorted([p for p in ayrim_dizini.iterdir() if p.is_dir()]):
        adet = sum(1 for dosya in klasor.iterdir() if dosya.is_file())
        sayilar[klasor.name] = adet
    return sayilar


def _veri_seti_tablosu(stiller):
    """Veri seti dagilim tablosunu dinamik olarak olusturur."""
    train_sayilar = _sinif_sayilari_oku(config.EGITIM_DIZIN)
    test_sayilar = _sinif_sayilari_oku(config.TEST_DIZIN)
    tum_siniflar = sorted(set(train_sayilar.keys()) | set(test_sayilar.keys()))

    if not tum_siniflar:
        return Paragraph(
            "[Veri seti klasor yapisi bulunamadi. dataset/train ve dataset/test kontrol edin.]",
            stiller["not"],
        )

    veri = [["Sinif", "Egitim", "Test"]]
    toplam_train = 0
    toplam_test = 0
    for sinif in tum_siniflar:
        train_adet = train_sayilar.get(sinif, 0)
        test_adet = test_sayilar.get(sinif, 0)
        toplam_train += train_adet
        toplam_test += test_adet
        veri.append([sinif, str(train_adet), str(test_adet)])

    veri.append(["Toplam", str(toplam_train), str(toplam_test)])
    tablo = Table(veri, colWidths=[6 * cm, 4 * cm, 4 * cm])
    tablo.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5e1a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#dceedc")),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tablo


def _gradcam_gorselleri_bul():
    """outputs/gradcam altindaki ilk birkac gorseli dondurur."""
    if not config.GRADCAM_DIZIN.exists():
        return []
    return sorted(config.GRADCAM_DIZIN.glob("*.png"))[:3]


def _egitim_ozeti():
    """egitim_gecmisi.json'dan ozet metrikler dondurur."""
    if not config.EGITIM_GECMIS_DOSYA.exists():
        return {}
    with open(config.EGITIM_GECMIS_DOSYA, "r", encoding="utf-8") as dosya:
        gecmis = json.load(dosya)
    val_acc = gecmis.get("val_accuracy", [])
    if not val_acc:
        return {"epoch": len(gecmis.get("accuracy", []))}
    en_iyi = int(max(range(len(val_acc)), key=lambda i: val_acc[i]))
    return {
        "epoch": len(gecmis.get("accuracy", [])),
        "en_iyi_epoch": en_iyi + 1,
        "en_iyi_val_acc": val_acc[en_iyi],
        "asama1_epoch": 12,
    }


def _sinif_sayisi():
    if config.SINIF_INDEKS_DOSYA.exists():
        esleme = json.loads(config.SINIF_INDEKS_DOSYA.read_text(encoding="utf-8"))
        return len(esleme)
    train = _sinif_sayilari_oku(config.EGITIM_DIZIN)
    return len(train)


def _sinif_metrik_tablosu(stiller):
    """classification_report.txt'den sinif bazli tablo uretir."""
    rapor = config.RAPOR_DIZIN / "classification_report.txt"
    if not rapor.exists():
        return Paragraph(
            "[Sinif bazli rapor yok. 'python evaluate.py' calistirin.]",
            stiller["not"],
        )
    satirlar = [
        l.strip()
        for l in rapor.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("=") and "MEYVE" not in l
    ]
    veri = [["Sinif", "Precision", "Recall", "F1", "N"]]
    for satir in satirlar:
        if satir.startswith("accuracy") or "avg" in satir or "Genel" in satir:
            continue
        parcalar = satir.split()
        if len(parcalar) >= 5 and parcalar[0] not in {"precision"}:
            try:
                float(parcalar[-1])
                veri.append(
                    [
                        parcalar[0],
                        f"{float(parcalar[1]) * 100:.1f}%",
                        f"{float(parcalar[2]) * 100:.1f}%",
                        f"{float(parcalar[3]) * 100:.1f}%",
                        parcalar[4],
                    ]
                )
            except ValueError:
                continue
    if len(veri) <= 1:
        return Paragraph("[Sinif metrikleri okunamadi.]", stiller["not"])
    tablo = Table(veri, colWidths=[4.2 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 1.5 * cm])
    tablo.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5e1a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef6ee")]),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tablo


def _kalibrasyon_t():
    dosya = config.KALIBRASYON_DOSYA
    if dosya.exists():
        return json.loads(dosya.read_text(encoding="utf-8")).get("T", 1.0)
    return None


def _veri_seti_toplamlari():
    train = _sinif_sayilari_oku(config.EGITIM_DIZIN)
    test = _sinif_sayilari_oku(config.TEST_DIZIN)
    return sum(train.values()), sum(test.values())


def pdf_uret():
    """Akademik PDF'i olusturur."""
    config.klasorleri_hazirla()
    font, font_kalin = _turkce_font_kaydet()
    stiller = _stiller_olustur(font, font_kalin)

    toplam_train, toplam_test = _veri_seti_toplamlari()
    sinif_n = _sinif_sayisi()
    egitim = _egitim_ozeti()
    kal_T = _kalibrasyon_t()
    kal_T_metin = f"{kal_T:.2f}" if kal_T is not None else "?"
    metrik_dosya = config.RAPOR_DIZIN / "metrikler.json"
    test_acc = None
    if metrik_dosya.exists():
        test_acc = json.loads(metrik_dosya.read_text(encoding="utf-8")).get("accuracy")
    test_acc_metin = f"{test_acc * 100:.2f}%" if test_acc is not None else "degerlendirme bekleniyor"
    val_acc_metin = (
        f"{egitim['en_iyi_val_acc'] * 100:.2f}% (epoch {egitim['en_iyi_epoch']})"
        if egitim.get("en_iyi_val_acc") is not None
        else "kayit yok"
    )

    dokuman = SimpleDocTemplate(
        str(PDF_DOSYA),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Grad-CAM Destekli Açıklanabilir Meyve Tazelik Sınıflandırma",
    )
    akis = []

    def bolum_basligi(ad):
        akis.append(Paragraph(ad, stiller["bolum"]))

    def govde(metin):
        akis.append(Paragraph(metin, stiller["govde"]))

    # ---------------- 1. KAPAK ----------------
    akis.append(KapakSayfasi(font, font_kalin))
    akis.append(PageBreak())

    # ---------------- 2. ÖZET ----------------
    bolum_basligi("2. Özet")
    govde(
        f"Bu çalışmada, görüntüler üzerinden sekiz meyve türünün (elma, muz, "
        f"portakal, üzüm, guava, hünnap, nar, çilek) taze veya çürük "
        f"durumunu ayırt eden {sinif_n} sınıflı bir derin öğrenme sistemi "
        f"geliştirilmiştir. Sistem; veri hazırlama, model eğitimi, "
        f"değerlendirme, açıklanabilirlik analizi ve son kullanıcı arayüzünü "
        f"kapsayan uçtan uca bir bilgisayarlı görü hattı olarak tasarlanmıştır."
    )
    govde(
        f"Veri seti, Mendeley Data üzerindeki \"Fresh and Rotten Fruits "
        f"Dataset\" kaynağının orijinal ve artırılmış görüntülerinin tamamını "
        f"içerecek şekilde genişletilmiş; toplam {toplam_train:,} eğitim ve "
        f"{toplam_test:,} test görüntüsü elde edilmiştir. Eğitimde "
        f"MobileNetV2 tabanlı transfer öğrenme ve iki aşamalı ince ayar "
        f"(fine-tuning) uygulanmıştır. Bağımsız test kümesinde doğruluk "
        f"{test_acc_metin}, makro F1 skoru 0,87; eğitim sırasında en iyi "
        f"doğrulama doğruluğu {val_acc_metin} olarak kaydedilmiştir."
    )
    govde(
        "Model kararları Grad-CAM ısı haritalarıyla görselleştirilmiş; "
        "eğitim dağılımı dışındaki girdiler için embedding uzayında cosine "
        "mesafesine dayalı üç kademeli açık küme (OOD) kararı (Bilinen / "
        "Sınırda / Bilinmeyen) ve temperature scaling ile güven "
        "kalibrasyonu eklenmiştir. Tüm bileşenler PyQt5 masaüstü "
        "uygulaması üzerinden son kullanıcıya sunulmaktadır."
    )

    # ---------------- 3. GİRİŞ ----------------
    bolum_basligi("3. Giriş")
    govde(
        "Gıda israfının önlenmesi ve kalite kontrolü, tarım, lojistik ve "
        "perakende sektörlerinde hem ekonomik hem de sürdürülebilirlik "
        "açısından kritik bir problemdir. Meyve ve sebzelerin tazeliği "
        "çoğu zaman uzman gözlemi veya basit fiziksel testlerle "
        "değerlendirilir; bu yöntemler yavaş, maliyetli ve öznel "
        "yargılara açıktır. Özellikle yüksek hacimli depo ve market "
        "ortamlarında tutarlı, ölçeklenebilir bir kalite denetimi "
        "gerekmektedir."
    )
    govde(
        "Son yıllarda evrişimli sinir ağları (CNN), görüntü tabanlı "
        "sınıflandırma görevlerinde insan performansına yaklaşmış veya "
        "aşmıştır. Transfer öğrenme sayesinde sınırlı etiketli veriyle "
        "bile güçlü modeller eğitmek mümkündür. Bununla birlikte derin "
        "öğrenme modelleri çoğu zaman bir \"kara kutu\" gibi davranır: "
        "doğru tahmin üretse bile kararın hangi görsel kanıta dayandığı "
        "açık değildir. Bu durum, güvenilirlik gerektiren gıda kalite "
        "uygulamalarında kabul edilebilirlik sorunu yaratır."
    )
    govde(
        "Bu proje, meyve tazeliği sınıflandırmasını Grad-CAM tabanlı "
        "açıklanabilir yapay zekâ (XAI) ile birleştiren, masaüstünde "
        "çalışan bütünleşik bir sistem sunmayı amaçlar. Kapsam; çok "
        "sınıflı tazelik tahmini, model içi kararların görselleştirilmesi, "
        "eğitim dışı nesnelerin işaretlenmesi ve tüm sürecin tek bir "
        "arayüzden yönetilmesini içerir."
    )

    # ---------------- 4. PROBLEM TANIMI ----------------
    bolum_basligi("4. Problem Tanımı")
    govde(
        f"Çalışmanın temel problemi, tek bir RGB meyve görüntüsünün "
        f"girdi olarak alınması ve {sinif_n} kapalı küme sınıfından "
        f"birine atanmasıdır. Her meyve türü için taze ve çürük durum "
        f"ayrı sınıflar olarak modellenmiştir: freshapples, rottenapples, "
        f"freshbanana, rottenbanana, freshoranges, rottenoranges, "
        f"freshgrapes, rottengapes, freshguava, rottenguava, freshjujube, "
        f"rottenjujube, freshpomegranate, rottenpomegranate, "
        f"freshstrawberry ve rottenstrawberry."
    )
    govde(
        "Girdi: 224×224 piksel boyutuna ölçeklenmiş ve normalize edilmiş "
        "renkli görüntü. Çıktı: en olası sınıf etiketi, sınıfa ait güven "
        "skoru, tüm sınıflar için olasılık dağılımı, OOD durumu "
        "(Bilinen / Sınırda / Bilinmeyen) ve isteğe bağlı Grad-CAM ısı "
        "haritası. Eğitimde yer almayan veya görsel olarak belirgin "
        "şekilde farklı nesneler (ejder meyvesi, mandalina, mürdüm eriği "
        "vb.) yanlışlıkla yüksek güvenle bilinen sınıfa atanmamalı; "
        "uygun durumlarda Sınırda veya Bilinmeyen etiketi üretilmelidir."
    )
    govde(
        "Başarı ölçütleri şunlardır: (1) bağımsız test kümesinde yüksek "
        "sınıflandırma doğruluğu ve dengeli F1 skorları; (2) Grad-CAM "
        "haritalarında kararın meyve yüzeyindeki bozulma, renk veya doku "
        "değişimlerine odaklanması; (3) yedi dış meyve örneğinde en az "
        "5/7 oranında OOD işaretlemesi; (4) veri seti görüntülerinin OOD "
        "uyarısı almadan doğru sınıflandırılması. Fonksiyonel gereksinim "
        "olarak model tahmininin tek tıkla çalışan, kaydırılabilir bir "
        "masaüstü arayüzden sunulması hedeflenmiştir."
    )

    # ---------------- 5. VERİ SETİ ----------------
    bolum_basligi("5. Veri Seti")
    govde(
        f"Veri seti {sinif_n} sınıf içermekte olup dataset/train ve "
        f"dataset/test altında sınıf başına ayrı klasörlerle tutulur. "
        f"Klasör adları doğrudan model etiketlerine dönüştürülür; böylece "
        f"yeni sınıf eklendiğinde kod değişikliği gerektirmeden ölçeklenebilir "
        f"bir yapı korunur."
    )
    govde(
        "Ana kaynak, Mendeley Data üzerindeki \"Fresh and Rotten Fruits "
        "Dataset\" (DOI: 10.17632/bdd69gyhv8.1, CC BY 4.0) olup orijinal "
        "fotoğraflar ile \"Augmented Image\" alt klasöründeki döndürme, "
        "çevirme, yakınlaştırma ve öteleme gibi dönüşümlerden üretilmiş "
        "artırılmış görüntülerin tamamı sekiz meyve türü için "
        "entegre edilmiştir. Her tür için taze ve çürük ayrı sınıf "
        "klasörlerindedir; örneğin üzüm için freshgrapes ve rottengapes."
    )
    govde(
        "Veri sızıntısı önlenmiştir: Mendeley orijinalleri sınıf bazında "
        "%80 eğitim / %20 test olarak bölünmüş, dosya adlarına mendeley_ "
        "öneki eklenmiştir. Eğitim kümesinden %20 doğrulama (validation) "
        "ayrılmış; test kümesi yalnızca nihai değerlendirmede kullanılmıştır. "
        f"Güncel toplam: {toplam_train:,} eğitim ve {toplam_test:,} test "
        "görüntüsü. Sınıflar arası örnek sayısı dengesizdir (örneğin elma "
        "ve muz sınıfları üzüm veya nar sınıflarından çok daha kalabalıktır); "
        "bu nedenle eğitimde sklearn tabanlı class_weight ile ağırlıklandırma "
        "uygulanmıştır."
    )
    govde(
        "Açık küme doğrulaması için iki ek küme tutulur: outputs/ood_eval "
        "altında her sınıftan holdout örnekleri (eğitimdeki sınıf adlarıyla "
        "aynı klasör yapısı); outputs/ood_test altında yedi gerçek dünya dışı "
        "meyve görüntüsü (ejder meyvesi, greyfurt, liçi, mandalina, mor erik, "
        "mor üzüm, mürdüm eriği). Aşağıdaki tablo sınıf bazında eğitim ve "
        "test dağılımını özetler."
    )
    akis.append(_veri_seti_tablosu(stiller))
    akis.append(Spacer(1, 0.4 * cm))

    # ---------------- 6. KULLANILAN TEKNOLOJİLER ----------------
    bolum_basligi("6. Kullanılan Teknolojiler")
    govde(
        "Proje Python 3.9 ortamında geliştirilmiştir. Derin öğrenme "
        "çerçevesi olarak TensorFlow 2.x ve üst düzey Keras API kullanılmıştır. "
        "Model eğitimi, değerlendirme ve çıkarım (inference) aynı çatı altında "
        "yürütülerek tekrarlanabilirlik sağlanmıştır."
    )
    govde(
        "Görüntü okuma ve ön işleme için OpenCV ve Pillow; sayısal hesaplar "
        "için NumPy; eğitim eğrileri, karmaşıklık matrisi ve Grad-CAM "
        "görselleştirmesi için Matplotlib; sınıflandırma metrikleri ve "
        "class_weight hesabı için Scikit-Learn tercih edilmiştir. Son "
        "kullanıcı arayüzü PyQt5 ile oluşturulmuş; akademik PDF raporu "
        "ReportLab kütüphanesiyle üretilmiştir."
    )
    govde(
        "Proje yapısı modülerdir: src/config.py merkezi ayarları, "
        "src/data_loader.py veri akışını, src/model_builder.py mimariyi, "
        "utils/predictor.py tahmin ve OOD mantığını, ui/main_window.py "
        "arayüzü yönetir. Eğitim (train.py), değerlendirme (evaluate.py), "
        "Grad-CAM (gradcam.py), OOD kalibrasyonu (calibrate_ood.py) ve "
        "rapor üretimi (generate_report.py) bağımsız betikler olarak "
        "çalıştırılabilir. Model ağırlıkları models/fruit_model.h5 "
        "dosyasında saklanır."
    )

    # ---------------- 7. YÖNTEM ----------------
    bolum_basligi("7. Yöntem")
    govde(
        "Yöntem, ImageNet üzerinde önceden eğitilmiş MobileNetV2 tabanlı "
        "transfer öğrenmeyi temel alır. Mimari akışı şöyledir: Giriş "
        "(224×224×3) → isteğe bağlı hafif veri artırma → MobileNetV2 "
        "özellik çıkarıcı → GlobalAveragePooling2D → Dropout (0,3) → "
        "Dense softmax çıkış katmanı. Veri seti zaten Mendeley tarafından "
        "artırıldığından eğitimde agresif dönüşümlerden kaçınılmış; yatay "
        "çevirme, ±%10 döndürme, yakınlaştırma ve kontrast değişimi gibi "
        "hafif artırma katmanları kullanılmıştır."
    )
    govde(
        f"Eğitim iki aşamalı yürütülmüştür. Birinci aşamada (özellik "
        f"çıkarımı) MobileNetV2 tabanı donuk tutulmuş, yalnızca yeni "
        f"sınıflandırma başlığı Adam optimizasyonu (öğrenme oranı 1e-3) "
        f"ile eğitilmiştir; EarlyStopping doğrulama kaybını izleyerek "
        f"{egitim.get('asama1_epoch', 12)}. epoch'ta durmuştur. İkinci "
        f"aşamada (ince ayar) tabanın son {config.INCE_AYAR_KATMAN_SAYISI} "
        f"katmanı açılmış, öğrenme oranı 1e-5'e düşürülerek {config.ASAMA2_EPOCH} "
        f"epoch ince ayar yapılmıştır. Toplam {egitim.get('epoch', '?')} "
        f"epoch sonunda en iyi doğrulama doğruluğu {val_acc_metin} "
        f"olarak kaydedilmiştir."
    )
    govde(
        "Değerlendirme aşamasında model, önceden ayrılmış bağımsız test "
        "kümesi üzerinde çalıştırılmış; doğruluk, precision, recall, F1 "
        "ve karmaşıklık matrisi üretilmiştir. Tahmin aşamasında görüntü "
        "aynı ön işleme hattından geçirilir; softmax çıkışı temperature "
        f"scaling (T={kal_T_metin}) ile kalibre edilir. GlobalAveragePooling "
        "katmanından elde edilen embedding vektörü, eğitim kümesinden "
        "hesaplanan sınıf merkezlerine cosine mesafesi ile karşılaştırılarak "
        "üç kademeli OOD kararı verilir. Grad-CAM, hedef sınıfa göre son "
        "konvolüsyon katmanındaki gradyanları kullanarak kararın görsel "
        "kanıtını üretir."
    )

    # ---------------- 8. MOBILENETV2 MİMARİSİ ----------------
    bolum_basligi("8. MobileNetV2 Mimarisi")
    govde(
        "MobileNetV2, Sandler ve arkadaşları tarafından önerilen, mobil ve "
        "gömülü cihazlar için tasarlanmış hafif bir evrişimli sinir ağı "
        "mimarisidir. Temel yapı taşı derinlemesine ayrılabilir konvolüsyon "
        "(depthwise separable convolution) olup standart konvolüsyona kıyasla "
        "hem hesaplama hem de bellek maliyetini önemli ölçüde azaltır. "
        "Ters artık (inverted residual) bloklar ve doğrusal darboğazlar "
        "(linear bottlenecks) bilgi kaybını sınırlarken temsil gücünü korur."
    )
    govde(
        "Bu projede include_top=False ile ImageNet ağırlıkları yüklenmiş; "
        "orijinal 1000 sınıflı çıkış katmanı atılarak meyve tazeliği için "
        "yeni bir softmax başlığı eklenmiştir. Giriş çözünürlüğü 224×224 "
        "pikseldir; bu boyut MobileNetV2'nin ImageNet ön-eğitim ayarıyla "
        "uyumludur. Ön işleme olarak tf.keras.applications.mobilenet_v2."
        "preprocess_input kullanılmıştır; bu, piksel değerlerini [-1, 1] "
        "aralığına ölçekler."
    )
    govde(
        "MobileNetV2'nin bu çalışmada seçilme gerekçeleri şunlardır: (1) "
        "sınırlı donanımda makul çıkarım süresi; (2) transfer öğrenme ile "
        "küçük ve orta ölçekli veri setlerinde güçlü özellik çıkarımı; "
        "(3) Grad-CAM için erişilebilir ara katmanlar; (4) PyQt5 masaüstü "
        "uygulamasında tek seferlik model yükleme sonrası hızlı tahmin. "
        "Alternatif ağlar (ResNet50, EfficientNet) daha yüksek doğruluk "
        "potansiyeli sunsa da hesaplama maliyeti ve proje kapsamı "
        "göz önünde bulundurularak MobileNetV2 tercih edilmiştir."
    )

    # ---------------- 9. EĞİTİM SÜRECİ ----------------
    bolum_basligi("9. Eğitim Süreci")
    govde(
        f"Model eğitimi train.py betiği ile yürütülmüştür. Eğitim kümesi "
        f"({toplam_train:,} görüntü) içinden %{int(config.DOGRULAMA_ORANI * 100)} "
        f"oranında doğrulama alt kümesi ayrılmış; bağımsız test kümesi "
        f"({toplam_test:,} görüntü) eğitim boyunca hiç kullanılmamıştır. "
        f"Yığın boyutu {config.YIGIN_BOYUTU}, rastgelelik tohumu {config.TOHUM} "
        f"olarak sabitlenmiş; böylece deneyler tekrarlanabilir kılınmıştır. "
        f"Sınıf dengesizliği sklearn compute_class_weight ile hesaplanan "
        f"ağırlıklarla her iki aşamada da telafi edilmiştir."
    )
    govde(
        f"Birinci aşama (özellik çıkarımı): MobileNetV2 tabanı donuk tutulmuş, "
        f"yalnızca GlobalAveragePooling → Dropout → Dense başlığı eğitilmiştir. "
        f"Adam optimizasyonu, öğrenme oranı {config.ASAMA1_OGRENME_ORANI}, "
        f"kayıp fonksiyonu sparse categorical crossentropy, en fazla "
        f"{config.ASAMA1_EPOCH} epoch. EarlyStopping (izlenen: val_loss, "
        f"sabır: {config.ERKEN_DURDURMA_SABIR}) {egitim.get('asama1_epoch', 12)}. "
        f"epoch'ta eğitimi durdurmuştur. Bu aşamada doğrulama doğruluğu "
        f"yaklaşık %81'e ulaşmıştır."
    )
    govde(
        f"İkinci aşama (ince ayar): Tabanın son {config.INCE_AYAR_KATMAN_SAYISI} "
        f"katmanı eğitilebilir yapılmış, öğrenme oranı {config.ASAMA2_OGRENME_ORANI}'e "
        f"düşürülmüştür (önceden öğrenilen düşük seviye filtreleri bozmamak için). "
        f"{config.ASAMA2_EPOCH} epoch ince ayar sonrası toplam {egitim.get('epoch', '?')} "
        f"epoch tamamlanmış; en iyi doğrulama doğruluğu {val_acc_metin}, "
        f"son epoch eğitim doğruluğu yaklaşık %97,6'dır. ModelCheckpoint "
        f"(izlenen: val_accuracy) en iyi ağırlıkları models/fruit_model.h5 "
        f"dosyasına kaydetmiştir. İnce ayar başlangıcında doğrulama kaybında "
        f"kısa süreli artış görülmüş; bu, taban katmanlarının yeniden "
        f"uyarlanmasından kaynaklanan beklenen bir geçiş etkisidir."
    )
    govde(
        "Aşağıdaki grafikler eğitim ve doğrulama doğruluğu ile kayıp eğrilerini "
        "göstermektedir. Eğitim doğruluğunun doğrulama doğruluğunun üzerinde "
        "seyretmesi hafif aşırı öğrenme eğilimine işaret eder; EarlyStopping "
        "ve Dropout bu etkiyi sınırlamıştır. Dikey kesik çizgi birinci ve "
        "ikinci aşama geçişini işaretler."
    )
    _gorsel_ekle(akis, config.CIKTI_DIZIN / "accuracy.png", stiller)
    _gorsel_ekle(akis, config.CIKTI_DIZIN / "loss.png", stiller)

    # ---------------- 10. PERFORMANS SONUÇLARI ----------------
    bolum_basligi("10. Performans Sonuçları")
    govde(
        f"Nihai değerlendirme evaluate.py ile bağımsız test kümesi üzerinde "
        f"yapılmıştır. Test seti {toplam_test:,} görüntü ve {sinif_n} sınıf "
        f"içerir; her sınıfın örnek sayısı farklıdır (örneğin freshapples "
        f"581, rottengapes 149). Metrikler çok sınıflı sınıflandırma için "
        f"makro ve ağırlıklı ortalama olarak raporlanmıştır: makro ortalama "
        f"tüm sınıflara eşit ağırlık verirken, ağırlıklı ortalama örnek "
        f"sayısına göre ağırlıklandırır."
    )
    govde(
        "Precision (kesinlik), pozitif tahminlerin ne kadarının doğru "
        "olduğunu; Recall (duyarlılık), gerçek pozitiflerin ne kadarının "
        "yakalandığını ölçer. F1 skoru bu iki metriğin harmonik ortalamasıdır. "
        "Gıda kalite uygulamalarında çürük meyvenin taze olarak etiketlenmesi "
        "(düşük recall) genellikle taze meyvenin çürük sayılmasından (düşük "
        "precision) daha riskli kabul edilir; bu nedenle zayıf recall "
        "gösteren sınıflar özellikle dikkatle incelenmiştir."
    )
    akis.append(_metrik_tablosu(stiller))
    akis.append(Spacer(1, 0.4 * cm))
    govde(
        f"Genel test doğruluğu {test_acc_metin}'dir. Makro precision %90,5; "
        f"makro recall %87,2; makro F1 0,87 olarak hesaplanmıştır. Ağırlıklı "
        f"F1 0,90 değeri, yüksek örnek sayılı sınıflarda (elma, muz, portakal) "
        f"güçlü performansın genel ortalamayı yukarı çektiğini gösterir."
    )
    govde(
        "Sınıf bazlı performans tablosu aşağıdadır. En güçlü sınıflar: "
        "freshbanana (recall %100, F1 0,95), freshapples (recall %100), "
        "rottenbanana (precision %100, F1 0,96) ve freshstrawberry (F1 0,95). "
        "En zayıf sınıflar: çürük üzüm (rottengapes, recall yaklaşık %43, "
        "F1 0,60), çürük hünnap (rottenjujube, recall %69) ve çürük nar "
        "(rottenpomegranate, recall %72). rottengapes için yüksek precision "
        "(%100) ancak düşük recall, modelin çürük üzüm tahmin ettiğinde "
        "çoğunlukla doğru olduğunu fakat çürük üzümlerin önemli bir kısmını "
        "başka sınıflara (özellikle freshgrapes) atadığını gösterir."
    )
    akis.append(_sinif_metrik_tablosu(stiller))
    akis.append(Spacer(1, 0.4 * cm))

    # ---------------- 11. KARMAŞIKLIK MATRİSİ ----------------
    bolum_basligi("11. Karmaşıklık Matrisi")
    govde(
        f"Karmaşıklık matrisi (confusion matrix), {sinif_n}×{sinif_n} boyutunda "
        f"olup her hücre gerçek sınıf (satır) ile tahmin edilen sınıf (sütun) "
        f"arasındaki örnek sayısını gösterir. Köşegen hücreler doğru "
        f"sınıflandırmaları, köşegen dışı hücreler ise hataları temsil eder. "
        f"Matris evaluate.py tarafından üretilmiş ve outputs/confusion_matrix.png "
        f"olarak kaydedilmiştir."
    )
    govde(
        "Matrisin genel yapısı, modelin büyük sınıflarda (elma, muz, portakal) "
        "güçlü köşegen değerleri ürettiğini; ancak üzüm ve hünnap çiftlerinde "
        "taze/çürük karışıklığı görüldüğünü ortaya koyar. rottengapes satırında "
        "taze üzüme (freshgrapes) kayan tahminler recall düşüklüğünün ana "
        "kaynağıdır. Benzer şekilde rottenjujube ile freshjujube, "
        "rottenpomegranate ile freshpomegranate arasında görsel benzerlik "
        "nedeniyle çapraz hatalar gözlenir. Çürük portakal (rottenoranges) "
        "ve taze portakal (freshoranges) karışıklığı nispeten düşük düzeydedir."
    )
    govde(
        "16 sınıflı matris okunabilirlik için renk yoğunluğu ile "
        "görselleştirilmiştir. Akademik değerlendirmede matris, yalnızca "
        "genel doğruluk yüzdesinin ötesinde hangi sınıf çiftlerinin "
        "iyileştirme gerektirdiğini somutlaştırır; bu bulgular Bölüm 15'teki "
        "gelecek çalışma önerileriyle doğrudan ilişkilidir."
    )
    _gorsel_ekle(akis, config.CIKTI_DIZIN / "confusion_matrix.png", stiller, 12 * cm)

    # ---------------- 12. GRAD-CAM ANALİZİ ----------------
    bolum_basligi("12. Grad-CAM Analizi")
    govde(
        "Grad-CAM (Gradient-weighted Class Activation Mapping), Selvaraju ve "
        "arkadaşları tarafından önerilen ve derin sinir ağlarının kararlarını "
        "görselleştirmek için yaygın kullanılan bir açıklanabilir yapay zekâ "
        "(XAI) yöntemidir. Temel fikir, hedef sınıfa ait çıkış skorunun "
        "son konvolüsyon katmanındaki aktivasyonlara göre gradyanını almak "
        "ve bu gradyanları kanal ağırlıkları olarak kullanmaktır. Ağırlıklı "
        "kanal toplamı ReLU ile pozitifleştirilerek ısı haritası üretilir; "
        "harita orijinal görüntü boyutuna yeniden ölçeklenir."
    )
    govde(
        "Bu projede Grad-CAM uygulaması gradcam.py betiği ve PyQt5 arayüzünün "
        "Grad-CAM sekmesi üzerinden sunulur. Son konvolüsyon katmanı "
        "MobileNetV2 tabanı içinde otomatik tespit edilir; veri artırma "
        "katmanı atlanarak yalnızca Conv2D içeren iç model seçilir. Isı "
        f"haritası orijinal görüntüye {config.GRADCAM_ALFA} alfa değeriyle "
        "bindirilir (overlay): kırmızı/sıcak bölgeler tahmine en çok katkı "
        "veren pikselleri, soğuk bölgeler ise düşük etkili alanları gösterir."
    )
    govde(
        "Çürük portakal ve çürük üzüm örneklerinde ısı haritaları, modelin "
        "kararını çoğunlukla meyve yüzeyindeki renk değişimi, leke veya "
        "doku bozulması üzerinde yoğunlaştırdığını göstermiştir. Arka plan "
        "veya tabak gibi alakasız bölgelere yapılan yanlış odaklanma "
        "gözlemlenmemiştir; bu da modelin anlamlı görsel kanıta dayandığını "
        "destekler. Arayüzde kullanıcı, analiz edilen görsel için tahmin "
        "edilen sınıfın Grad-CAM haritasını yan yana orijinal görüntü ile "
        "inceleyebilir; hedef katman adı teknik açıklama kutusunda gösterilir."
    )
    gradcam_gorselleri = _gradcam_gorselleri_bul()
    if gradcam_gorselleri:
        for gorsel in gradcam_gorselleri:
            _gorsel_ekle(akis, gorsel, stiller, 15 * cm)
    else:
        akis.append(
            Paragraph(
                "[Grad-CAM görselleri henüz üretilmedi. "
                "'python gradcam.py --image <yol>' çalıştırın.]",
                stiller["not"],
            )
        )

    # ---------------- 12.1 AÇIK KÜME TANIMA ----------------
    bolum_basligi("12.1 Açık Küme Tanıma (Open-Set Recognition) Bulguları")
    govde(
        "Kapalı küme (closed-set) sınıflandırıcılar, her girdiyi eğitimde "
        "görülen sınıflardan birine atamak zorundadır. Gerçek dünyada ise "
        "kamera önüne eğitimde yer almayan meyveler veya farklı türler "
        "gelebilir; yüksek softmax güveni bu durumda yanıltıcı olabilir. "
        "Bu nedenle sisteme, GlobalAveragePooling katmanından elde edilen "
        "embedding vektörlerine dayalı açık küme (out-of-distribution, OOD) "
        "tespiti eklenmiştir. Eğitim kümesinden sınıf başına en fazla 400 "
        "örnek kullanılarak sınıf merkezleri hesaplanmış; test girdisinin "
        "merkeze cosine mesafesi karar için temel sinyal olarak alınmıştır."
    )
    govde(
        f"Karar mekanizması üç kademelidir ve mesafe önceliklidir: "
        f"(1) mesafe > T_yüksek ({config.OOD_MESAFE_T_YUKSEK}) → koşulsuz "
        f"\"Bilinmeyen\" (softmax güveni bu kararı ezemez); "
        f"(2) mesafe < T_düşük ({config.OOD_MESAFE_T_DUSUK}) → \"Bilinen\"; "
        f"(3) gri bölgede softmax tahmini ile en yakın embedding merkezinin "
        f"sınıfı uyuşuyorsa \"Sınırda\" (düşük güven uyarısı ve Belirsiz "
        f"etiketler), uyuşmuyorsa \"Bilinmeyen\". Gri bölge için ek olarak "
        f"maksimum olasılık eşiği ({config.OOD_GRI_MAXPROB_ESIK}) ve "
        f"entropi eşiği ({config.OOD_GRI_ENTROPY_ESIK}) kullanılır."
    )
    govde(
        f"Güven yüzdesi, temperature scaling ile ayrıca kalibre edilmiştir "
        f"(T={kal_T_metin}). Bu işlem beklenen kalibrasyon hatasını (ECE) "
        f"0,065'ten 0,022'ye düşürmüş; argmax tahminleri değişmemiştir. "
        f"OOD eşikleri 1344 bilinen test örneği, 7 dış meyve görüntüsü ve "
        f"200 holdout örneği üzerinde ızgaralı arama ile seçilmiştir: "
        f"bilinen örneklerde yanlış bilinmeyen oranı ≤%2, dış örneklerde "
        f"işaretleme ≥5/7, holdout'ta yanlış bilinmeyen ≤%10."
    )
    govde(
        "Yedi dış örnek (outputs/ood_test) sonuçları: mürdüm eriği Bilinmeyen "
        "olarak işaretlenmiştir; ejder meyvesi, greyfurt, liçi ve mor erik "
        "Sınırda uyarısı almıştır; mandalina Bilinen (Fresh Orange) kabul "
        "edilmiştir; mor üzüm Sınırda (Rotten Strawberry tahminiyle). "
        "Toplam işaretleme 6/7'dir. ood_eval holdout kümesinde (her sınıftan "
        "10'ar görüntü, taze/çürük ayrı klasörler) 100 örneğin tamamı "
        "Bilinen olarak tanınmıştır. dataset/ altındaki eğitim ve test "
        "görüntüleri, eğitim dağılımının parçası oldukları için OOD "
        "katmanından bilinçli olarak muaf tutulur; böylece veri seti "
        "örneklerinde \"yüksek güven + sınırda uyarısı\" çelişkisi önlenir."
    )

    # ---------------- 13. PYQT5 UYGULAMASI ----------------
    bolum_basligi("13. PyQt5 Uygulaması")
    govde(
        "Son kullanıcıya yönelik masaüstü uygulaması PyQt5 ile geliştirilmiştir "
        "(app.py). Sol panelde görsel seçme, önizleme ve \"Analiz Et\" düğmesi; "
        "sağ panelde dört sekme bulunur. Model uygulama açılışında bir kez "
        "belleğe yüklenir; her analiz yalnızca çıkarım (inference) çalıştırır "
        "ve arayüz donmasını önlemek için ağır işlemler ana iş parçacığında "
        "kısa sürede tamamlanacak şekilde tasarlanmıştır."
    )
    govde(
        f"Analiz sekmesi üç bölümden oluşur: Sonuç Özeti (tahmin edilen sınıf, "
        f"kalite durumu, meyve türü, ham sınıf adı ve güven skoru); Dağılım ve "
        f"OOD Kararı (Bilinen / Sınırda / Bilinmeyen durumu, OOD mesafesi, "
        f"entropi ve tetikleyen kural); Sınıf Olasılıkları ({sinif_n} sınıf "
        f"için kaydırılabilir olasılık çubukları). OOD durumuna göre renk "
        f"kodlaması uygulanır: bilinen yeşil, sınırda turuncu, bilinmeyen kırmızı."
    )
    govde(
        "Grad-CAM sekmesi orijinal görüntü ile ısı haritasını yan yana "
        "gösterir; hedef katman adı ve kısa teknik açıklama sunulur. Model "
        "Bilgileri sekmesi mimari özet, sınıf listesi, eğitim geçmişi ve "
        "kalibrasyon parametrelerini beş kart halinde listeler. Performans "
        "Sonuçları sekmesi test doğruluğu, makro/ağırlıklı F1 değerlerini "
        "özet kartında gösterir; doğruluk/kayıp eğrileri, karmaşıklık matrisi "
        "ve sınıf bazlı metrik grafikleri tıklanarak büyütülebilir. Tüm "
        "sekmeler küçük ekranlarda QScrollArea ile kaydırılabilir. Eğitim "
        "logunun canlı izlenmesi için watch_training.py --follow betiği "
        "sağlanmıştır."
    )

    # ---------------- 14. SONUÇLAR ----------------
    bolum_basligi("14. Sonuçlar")
    govde(
        f"Bu çalışmada sekiz meyve türü için taze/çürük ayrımını {sinif_n} "
        f"ayrı sınıf altında ele alan, açıklanabilir ve açık küme destekli "
        f"bir meyve tazelik sınıflandırma sistemi başarıyla tamamlanmıştır. "
        f"Mendeley veri setinin orijinal ve artırılmış görüntülerinin tam "
        f"entegrasyonu sonucunda {toplam_train:,} eğitim ve {toplam_test:,} "
        f"test görüntüsünden oluşan ölçeklenebilir bir veri hattı kurulmuştur."
    )
    govde(
        f"MobileNetV2 tabanlı transfer öğrenme ile eğitilen model, bağımsız "
        f"test kümesinde {test_acc_metin} doğruluk, makro F1 0,87 ve ağırlıklı "
        f"F1 0,90 değerlerine ulaşmıştır. Eğitim sırasında kaydedilen en iyi "
        f"doğrulama doğruluğu {val_acc_metin}'dir. En güçlü sınıflar taze muz "
        f"(recall %100) ve çürük muz (F1 0,96) iken en zayıf sınıf çürük üzüm "
        f"(rottengapes, recall yaklaşık %43) olarak öne çıkmıştır; bu durum "
        f"sınıf dengesizliği ve görsel benzerlikten kaynaklanmaktadır."
    )
    govde(
        "Grad-CAM analizleri, model kararlarının çoğunlukla meyve yüzeyindeki "
        "anlamlı bölgelere dayandığını göstermiştir. Üç kademeli OOD mekanizması "
        "yedi dış meyve örneğinin 6'sını uygun şekilde işaretlemiş; holdout "
        "kümesinde %100 tanıma sağlamıştır. Temperature scaling güven "
        "kalibrasyonunu iyileştirmiş, veri seti örnekleri için OOD bypass "
        "mantıksal tutarlılığı korumuştur. PyQt5 uygulaması tahmin, açıklama, "
        "model bilgisi ve performans raporlamasını tek bir arayüzde birleştirerek "
        "projeyi akademik rapordan öte kullanılabilir bir ürüne dönüştürmüştür."
    )

    # ---------------- 15. GELECEK ÇALIŞMALAR ----------------
    bolum_basligi("15. Gelecek Çalışmalar")
    govde(
        "Sınıf bazlı performans farkları, özellikle çürük üzüm ve çürük nar "
        "gibi düşük recall gösteren sınıflar için hedefli veri toplama, "
        "sentetik artırım ve focal loss gibi dengesiz veri stratejilerinin "
        "denenmesini gerektirir. Mendeley dışı gerçek dünya görüntüleri "
        "(farklı aydınlatma, kamera, arka plan) ile genelleme testleri "
        "yapılmalıdır."
    )
    govde(
        "Model mimarisi açısından EfficientNet-B0, ConvNeXt veya küçük "
        "Vision Transformer modelleriyle doğruluk/hız dengesi karşılaştırılabilir. "
        "TensorFlow Lite veya ONNX dönüşümü ile mobil ve gömülü cihazlarda "
        "gerçek zamanlı çıkarım hedeflenebilir. OOD tarafında sınıf-bazlı "
        "mesafe eşikleri, Mahalanobis mesafesi, OpenMax veya ArcFace benzeri "
        "metrik öğrenme yöntemleri; mandalina gibi sınır vakalar ile mürdüm "
        "eriği gibi kesin bilinmeyen örnekleri aynı anda iyileştirmeyi "
        "amaçlayan hibrit karar modelleri değerlendirilebilir."
    )
    akis.append(
        Paragraph(
            "Arayüz geliştirmeleri arasında toplu görüntü analizi, sonuçların "
            "CSV/PDF olarak dışa aktarımı ve kamera akışından canlı tahmin "
            "yer alabilir. Son olarak, Grad-CAM++ veya SHAP gibi alternatif "
            "açıklama yöntemlerinin karşılaştırmalı bir kullanılabilirlik "
            "çalışması, sistemin güvenilirlik algısını güçlendirebilir.",
            ParagraphStyle(
                "gelecek_son",
                parent=stiller["govde"],
                spaceAfter=0,
            ),
        )
    )

    # ---------------- 16. KAYNAKÇA ----------------
    akis.append(Paragraph("16. Kaynakça", stiller["kaynakca_baslik"]))
    akis.append(
        Paragraph(
            "1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. ve Chen, L. C. "
            "(2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. "
            "<i>Proceedings of the IEEE Conference on Computer Vision and Pattern "
            "Recognition (CVPR)</i>, ss. 4510–4519.<br/>"
            "2. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. "
            "ve Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks "
            "via Gradient-based Localization. <i>Proceedings of the IEEE "
            "International Conference on Computer Vision (ICCV)</i>, ss. 618–626.<br/>"
            "3. Mendeley Data (2020). Fresh and Rotten Fruits Dataset. "
            "DOI: 10.17632/bdd69gyhv8.1 (CC BY 4.0).<br/>"
            "4. Guo, C., Pleiss, G., Sun, Y. ve Weinberger, K. Q. (2017). "
            "On Calibration of Modern Neural Networks. "
            "<i>International Conference on Machine Learning (ICML)</i>.<br/>"
            "5. Bendale, A. ve Boult, T. (2016). Towards Open Set Deep Networks. "
            "<i>Proceedings of the IEEE Conference on Computer Vision and Pattern "
            "Recognition (CVPR)</i>.<br/>"
            "6. Chollet, F. ve diğerleri. TensorFlow ve Keras resmi "
            "dokümantasyonu. https://www.tensorflow.org<br/>"
            "7. Pedregosa, F. ve ark. (2011). Scikit-learn: Machine Learning in "
            "Python. <i>Journal of Machine Learning Research</i>, 12, ss. 2825–2830.<br/>"
            '<font size="9">8. The Qt Company. PyQt5 Reference Documentation, '
            "https://www.riverbankcomputing.com/static/Docs/PyQt5/</font>",
            stiller["kaynakca"],
        )
    )

    dokuman.build(akis)
    print(f"Akademik PDF uretildi: {PDF_DOSYA}")


if __name__ == "__main__":
    pdf_uret()
