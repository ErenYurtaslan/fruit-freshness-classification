"""Ana masaustu arayuzu (son cilalama surumu).

Tasarim hedefi:
- Sade, modern, sicak ve guven veren bir kalite kontrol uygulamasi deneyimi
- Sekmeli yapida tahmin + Grad-CAM + model bilgisi + performans gorselleri
- Veri setinden hizli secim ve harici gorsel yukleme ile otomatik analiz akisi
"""

from __future__ import annotations

import random
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src import config
from utils.image_utils import (
    dosya_adini_kisa_goster,
    diziye_gore_pixmap_olustur,
    sinif_adini_kullanici_dostu_yap,
    yola_gore_pixmap_olustur,
)


class TiklanabilirGorselEtiketi(QLabel):
    """Performans sekmesindeki gorselleri tiklayarak buyutmek icin etiket."""

    def __init__(self, ebeveyn):
        super().__init__(ebeveyn)
        self.buyuk_goruntu_yolu = None

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton and self.buyuk_goruntu_yolu:
            pencere = self.window()
            if hasattr(pencere, "_gorseli_buyuk_ac"):
                pencere._gorseli_buyuk_ac(self.buyuk_goruntu_yolu)  # noqa: SLF001
        super().mousePressEvent(event)


class AnaPencere(QMainWindow):
    """Akıllı Meyve Kalite Kontrol Sistemi ana penceresi."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Akıllı Meyve Kalite Kontrol Sistemi")
        self.setMinimumSize(900, 620)
        self.resize(1280, 820)

        self.secili_goruntu_yolu = None
        self.tahmin_servisi = None
        self.gradcam_servisi = None
        self.prog_cubuklari = {}
        self.dataset_haritasi = {}
        self.dataset_duz_liste = {}
        self.dataset_tum_ornekler = []
        self._hizli_secenek_guncelleniyor = False
        self._performans_yollari = {}

        self._arayuzu_kur()
        self._stil_uygula()
        self._model_servislerini_yukle()
        self._dataset_hizli_secim_hazirla()
        # Yerlesim tamamlandiktan sonra performans gorsellerini hedef boyutta yukle.
        QTimer.singleShot(200, self._performans_gorsellerini_yukle)

    # ------------------------------------------------------------------
    # Kurulum
    # ------------------------------------------------------------------
    def _arayuzu_kur(self):
        merkez = QWidget()
        self.setCentralWidget(merkez)
        ana_dikey = QVBoxLayout(merkez)
        ana_dikey.setContentsMargins(20, 18, 20, 18)
        ana_dikey.setSpacing(14)

        baslik = QLabel("Akıllı Meyve Kalite Kontrol Sistemi")
        baslik.setObjectName("baslik")
        alt_baslik = QLabel("MobileNetV2 + Transfer Learning + Grad-CAM")
        alt_baslik.setObjectName("altBaslik")
        ana_dikey.addWidget(baslik)
        ana_dikey.addWidget(alt_baslik)

        govde = QHBoxLayout()
        govde.setSpacing(16)
        ana_dikey.addLayout(govde, stretch=1)

        govde.addWidget(self._kaydirilabilir_yap(self._sol_panel_olustur()), stretch=1)
        govde.addWidget(self._sekmeleri_olustur(), stretch=2)

    def _kaydirilabilir_yap(self, icerik: QWidget) -> QScrollArea:
        """Sekme veya panel icerigini dikey kaydirilabilir yapar."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(icerik)
        return scroll

    def _sol_panel_olustur(self) -> QWidget:
        panel = QWidget()
        sol = QVBoxLayout(panel)
        sol.setSpacing(12)

        grup = QGroupBox("Görüntü Yükleme")
        yerlesim = QVBoxLayout(grup)
        yerlesim.setSpacing(10)

        self.lbl_onizleme = QLabel("Henüz görsel seçilmedi")
        self.lbl_onizleme.setAlignment(Qt.AlignCenter)
        self.lbl_onizleme.setMinimumSize(260, 220)
        self.lbl_onizleme.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_onizleme.setFrameShape(QFrame.Box)
        self.lbl_onizleme.setObjectName("onizleme")

        self.lbl_dosya = QLabel("Dosya: -")
        self.lbl_dosya.setWordWrap(True)
        self.lbl_dosya.setObjectName("dosyaBilgi")

        # Harici gorsel yukleme
        self.btn_sec = QPushButton("Harici Görsel Yükle")
        self.btn_sec.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.btn_sec.clicked.connect(self._gorsel_sec)

        # Veri setinden hizli secim
        hizli_grup = QGroupBox("Veri Setinden Hızlı Seçim")
        hizli_form = QFormLayout(hizli_grup)
        hizli_form.setSpacing(8)
        hizli_form.setLabelAlignment(Qt.AlignLeft)

        self.cmb_bolum = QComboBox()
        self.cmb_bolum.addItems(["train", "test"])

        self.cmb_sinif = QComboBox()

        self.cmb_dosya = QComboBox()

        self.btn_hizli_yukle = QPushButton("Veri Setinden Yükle")
        self.btn_hizli_yukle.setIcon(
            self.style().standardIcon(QStyle.SP_DialogApplyButton)
        )
        self.btn_hizli_yukle.clicked.connect(self._hizli_secim_uygula)

        self.lbl_hizli_bilgi = QLabel(
            "Bu buton her tıklamada veri setinden rastgele bir görsel seçer."
        )
        self.lbl_hizli_bilgi.setWordWrap(True)
        self.lbl_hizli_bilgi.setObjectName("grafikAciklama")
        hizli_form.addRow(self.lbl_hizli_bilgi)
        hizli_form.addRow(self.btn_hizli_yukle)

        self.lbl_durum = QLabel("Model servisleri hazırlanıyor...")
        self.lbl_durum.setObjectName("durum")

        yerlesim.addWidget(self.lbl_onizleme)
        yerlesim.addWidget(self.lbl_dosya)
        yerlesim.addWidget(self.btn_sec)
        yerlesim.addWidget(hizli_grup)
        yerlesim.addWidget(self.lbl_durum)

        sol.addWidget(grup)
        return panel

    def _sekmeleri_olustur(self):
        self.sekmeler = QTabWidget()
        self.sekmeler.setMovable(False)
        self.sekme_analiz = self._sekme_analiz()
        self.sekme_gradcam = self._sekme_gradcam()
        self.sekme_model = self._sekme_model_bilgileri()
        self.sekme_performans = self._sekme_performans()

        self.sekmeler.addTab(self.sekme_analiz, "Analiz")
        self.sekmeler.addTab(self.sekme_gradcam, "Grad-CAM")
        self.sekmeler.addTab(self.sekme_model, "Model Bilgileri")
        self.sekmeler.addTab(self.sekme_performans, "Performans Sonuçları")

        # Az sayida ve sade ikonlar
        self.sekmeler.setTabIcon(
            0, self.style().standardIcon(QStyle.SP_FileDialogDetailedView)
        )
        self.sekmeler.setTabIcon(1, self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.sekmeler.setTabIcon(2, self.style().standardIcon(QStyle.SP_FileDialogInfoView))
        self.sekmeler.setTabIcon(3, self.style().standardIcon(QStyle.SP_DriveHDIcon))
        return self.sekmeler

    def _sekme_analiz(self):
        icerik = QWidget()
        ana = QVBoxLayout(icerik)
        ana.setSpacing(12)

        ozet = QGroupBox("Sonuç Özeti")
        ozet_y = QVBoxLayout(ozet)
        ozet_y.setSpacing(8)

        self.lbl_tahmin = QLabel("-")
        self.lbl_tahmin.setObjectName("tahminSinif")
        self.lbl_tahmin.setWordWrap(True)
        self.lbl_genel_durum = QLabel("Kalite Durumu: -")
        self.lbl_genel_durum.setObjectName("genelDurum")
        self.lbl_meyve_turu = QLabel("Meyve Türü: -")
        self.lbl_ham_sinif = QLabel("Tahmin Sınıfı: -")
        self.lbl_guven = QLabel("Güven Skoru: -")
        self.lbl_guven.setObjectName("kritikBilgi")
        self.lbl_guven.setWordWrap(True)

        ozet_y.addWidget(self.lbl_tahmin)
        ozet_y.addWidget(self.lbl_genel_durum)
        ozet_y.addWidget(self.lbl_meyve_turu)
        ozet_y.addWidget(self.lbl_ham_sinif)
        ozet_y.addWidget(self.lbl_guven)
        ana.addWidget(ozet)

        dagilim = QGroupBox("Dağılım ve OOD Kararı")
        dagilim_y = QVBoxLayout(dagilim)
        self.lbl_dagilim = QLabel("Dağılım Durumu: -")
        self.lbl_dagilim.setObjectName("kritikBilgi")
        self.lbl_dagilim.setWordWrap(True)
        self.lbl_dagilim_metrik = QLabel("OOD Skoru / Entropy / Mesafe: -")
        self.lbl_dagilim_metrik.setWordWrap(True)
        dagilim_y.addWidget(self.lbl_dagilim)
        dagilim_y.addWidget(self.lbl_dagilim_metrik)
        ana.addWidget(dagilim)

        olasilik_grup = QGroupBox("Sınıf Olasılıkları (16 sınıf)")
        olasilik_y = QVBoxLayout(olasilik_grup)
        olasilik_scroll = QScrollArea()
        olasilik_scroll.setWidgetResizable(True)
        olasilik_scroll.setFrameShape(QFrame.NoFrame)
        olasilik_scroll.setMinimumHeight(180)
        olasilik_icerik = QWidget()
        self.olasilik_grid = QGridLayout(olasilik_icerik)
        self.olasilik_grid.setHorizontalSpacing(12)
        self.olasilik_grid.setVerticalSpacing(6)
        self.olasilik_grid.setColumnStretch(0, 0)
        self.olasilik_grid.setColumnStretch(1, 1)
        olasilik_scroll.setWidget(olasilik_icerik)
        olasilik_y.addWidget(olasilik_scroll)
        ana.addWidget(olasilik_grup, stretch=1)

        return self._kaydirilabilir_yap(icerik)

    def _sekme_gradcam(self):
        icerik = QWidget()
        ana = QVBoxLayout(icerik)
        ana.setSpacing(12)

        baslik = QLabel(
            "Grad-CAM: modelin tahmin kararında hangi görüntü bölgelerine "
            "baktığını gösterir. Kırmızı/sıcak alanlar daha yüksek etkiye sahiptir."
        )
        baslik.setObjectName("sekmeBilgi")
        baslik.setWordWrap(True)
        ana.addWidget(baslik)

        self.lbl_grad_tahmin = QLabel("Hedef sınıf: Analiz bekleniyor")
        self.lbl_grad_tahmin.setObjectName("kritikBilgi")
        self.lbl_grad_tahmin.setWordWrap(True)
        ana.addWidget(self.lbl_grad_tahmin)

        gorsel_satir = QHBoxLayout()
        gorsel_satir.setSpacing(12)

        sol_grup = QGroupBox("Orijinal Görsel")
        sol_y = QVBoxLayout(sol_grup)
        self.lbl_grad_orijinal = QLabel("Görsel bekleniyor")
        self.lbl_grad_orijinal.setAlignment(Qt.AlignCenter)
        self.lbl_grad_orijinal.setMinimumSize(220, 180)
        self.lbl_grad_orijinal.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_grad_orijinal.setFrameShape(QFrame.Box)
        sol_y.addWidget(self.lbl_grad_orijinal)

        sag_grup = QGroupBox("Grad-CAM Isı Haritası")
        sag_y = QVBoxLayout(sag_grup)
        self.lbl_grad_overlay = QLabel("Analiz bekleniyor")
        self.lbl_grad_overlay.setAlignment(Qt.AlignCenter)
        self.lbl_grad_overlay.setMinimumSize(220, 180)
        self.lbl_grad_overlay.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lbl_grad_overlay.setFrameShape(QFrame.Box)
        sag_y.addWidget(self.lbl_grad_overlay)

        gorsel_satir.addWidget(sol_grup, stretch=1)
        gorsel_satir.addWidget(sag_grup, stretch=1)
        ana.addLayout(gorsel_satir, stretch=1)

        teknik = QGroupBox("Teknik Açıklama")
        teknik_y = QVBoxLayout(teknik)
        self.lbl_katman = QLabel("Hedef katman: -")
        self.lbl_katman.setWordWrap(True)
        self.lbl_grad_bilgi = QLabel(
            "Isı haritası, MobileNetV2'nin son konvolüsyon katmanındaki aktivasyon "
            "ağırlıklarından üretilir. Tahmin sınıfına en çok katkı veren pikseller "
            "vurgulanır."
        )
        self.lbl_grad_bilgi.setWordWrap(True)
        self.lbl_grad_bilgi.setObjectName("gradBilgi")
        teknik_y.addWidget(self.lbl_katman)
        teknik_y.addWidget(self.lbl_grad_bilgi)
        ana.addWidget(teknik)

        return self._kaydirilabilir_yap(icerik)

    def _sekme_model_bilgileri(self):
        icerik = QWidget()
        self.model_bilgi_layout = QVBoxLayout(icerik)
        self.model_bilgi_layout.setSpacing(12)
        return self._kaydirilabilir_yap(icerik)

    def _sekme_performans(self):
        icerik = QWidget()
        self.perf_layout = QVBoxLayout(icerik)
        self.perf_layout.setSpacing(12)

        ozet_grup = QGroupBox("Test Performans Özeti (16 sınıf)")
        ozet_y = QVBoxLayout(ozet_grup)
        self.lbl_perf_ozet = QLabel("Metrikler yükleniyor...")
        self.lbl_perf_ozet.setWordWrap(True)
        self.lbl_perf_ozet.setObjectName("grafikAciklama")
        ozet_y.addWidget(self.lbl_perf_ozet)
        self.perf_layout.addWidget(ozet_grup)

        self.lbl_acc_img = self._gorsel_karti_olustur(
            "1. Accuracy Eğrisi",
            "Epoch bazında eğitim ve doğrulama doğruluğu. Gri kesik çizgi fine-tuning "
            "(Aşama 2) başlangıcını gösterir. İki çizginin birlikte yükselmesi sağlıklı "
            "öğrenmeyi; büyük ayrışma overfit riskini işaret eder.",
        )
        self.lbl_loss_img = self._gorsel_karti_olustur(
            "2. Loss Eğrisi",
            "Eğitim ve doğrulama kaybının epoch boyunca değişimi. Fine-tuning "
            "başlangıcındaki geçici sıçrama normaldir; doğrulama loss'un sürekli "
            "artması ezberleme belirtisi olabilir.",
        )
        self.lbl_cm_img = self._gorsel_karti_olustur(
            "3. Confusion Matrix",
            "Sınıf bazında doğru/yanlış tahmin dağılımı (5.399 test görseli). "
            "Köşegen = doğru sınıflandırma. Özellikle çürük üzüm ve hünnap gibi "
            "düşük recall'lu sınıflar iyileştirme adayıdır.",
        )

        return self._kaydirilabilir_yap(icerik)

    def _gorsel_karti_olustur(self, baslik: str, aciklama: str):
        grup = QGroupBox(baslik)
        yer = QVBoxLayout(grup)
        bilgi = QLabel(aciklama)
        bilgi.setWordWrap(True)
        bilgi.setObjectName("grafikAciklama")
        lbl = TiklanabilirGorselEtiketi(self)
        lbl.setText("Görsel bulunamadı")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setMinimumHeight(260)
        lbl.setFrameShape(QFrame.Box)
        lbl.setToolTip("Büyütmek için tıklayın")
        yer.addWidget(bilgi)
        yer.addWidget(lbl)
        self.perf_layout.addWidget(grup)
        return lbl

    def _stil_uygula(self):
        """Sıcak ve modern tasarım paleti uygular.

        Palet:
        - Arka plan: antrasit + koyu indigo
        - Kartlar: krem/kırık beyaz/bej
        - Vurgu: pastel camgöbeği
        - Uyarı: bordo
        - Bilgi: taba/kahve
        """
        self.setStyleSheet(
            """
            QMainWindow { background-color: #7b4f36; }  /* tarcin */
            QWidget {
                color: #332a22;
                font-size: 12px;
                font-family: "Palatino Linotype", "Trebuchet MS", "Segoe UI";
            }

            QGroupBox {
                background-color: #f5ecdc;
                border: 1px solid #bfa98a;
                border-radius: 14px;
                margin-top: 14px;
                font-size: 13px;
                font-weight: 700;
                color: #4a3528;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 2px 10px;
                color: #4a3528;
                background-color: #ead8bc;
                border: 1px solid #c7ae89;
                border-radius: 7px;
            }

            #baslik {
                font-size: 26px;
                font-weight: 700;
                color: #f3ead8;
            }
            #altBaslik {
                font-size: 13px;
                color: #d6ccbb;
                margin-bottom: 6px;
            }
            #onizleme {
                border: 2px dashed #6b5b46;
                border-radius: 10px;
                background-color: #fff8eb;
                color: #6d5a4a;
            }
            #durum {
                color: #f2e9db;
                background-color: #4a3d30;
                border: 1px solid #72604f;
                border-radius: 9px;
                padding: 8px;
            }
            #dosyaBilgi {
                color: #efe8dc;
                background-color: #394067;
                border-radius: 8px;
                padding: 6px;
            }
            #tahminSinif {
                font-size: 34px;
                font-weight: 700;
                color: #4a3123;
            }
            #genelDurum {
                font-size: 22px;
                font-weight: 700;
            }
            #kritikBilgi {
                font-size: 16px;
                font-weight: 700;
                color: #3f2c20;
            }
            #sekmeBilgi {
                color: #4d3b2f;
                background-color: #efe1cc;
                border: 1px solid #d0b28e;
                border-radius: 8px;
                padding: 8px;
            }
            #gradBilgi {
                color: #5b4638;
                background-color: #f4e7d5;
                border: 1px solid #d9c2a4;
                border-radius: 8px;
                padding: 8px;
            }
            #grafikAciklama {
                color: #594637;
                background-color: #efe5d8;
                border: 1px solid #d8c4a8;
                border-radius: 8px;
                padding: 7px;
            }

            QPushButton {
                background-color: #4e567f; /* indigo */
                color: #f8f1e5;
                border-radius: 9px;
                padding: 9px 12px;
                font-weight: 600;
                border: 1px solid #7e8cb4;
            }
            QPushButton:hover:!disabled {
                background-color: #646f95;
            }
            QPushButton:disabled {
                background-color: #8a8fa4;
                color: #ded6ca;
            }

            QComboBox {
                background-color: #fff8eb;
                border: 1px solid #c2aa86;
                border-radius: 7px;
                padding: 6px;
            }
            QComboBox::drop-down { border: none; width: 20px; }

            QTabWidget::pane {
                border: 1px solid #75644f;
                border-radius: 10px;
                background: #f2e8d6;
                top: -1px;
            }
            QTabBar::tab {
                background: #cfbb9c;
                color: #4c392f;
                border: 1px solid #aa8f70;
                padding: 8px 14px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background: #96cdd2; /* pastel camgobegi sadece vurgu */
                color: #2d2d2d;
                border-color: #6baab0;
            }

            QProgressBar {
                border: 1px solid #c7ae8a;
                border-radius: 7px;
                text-align: center;
                background-color: #fff8eb;
                color: #3f352e;
            }
            QProgressBar::chunk {
                background-color: #74b7be; /* vurgu */
                border-radius: 6px;
            }

            QScrollArea { border: none; background: transparent; }
            QScrollArea > QWidget > QWidget { background: transparent; }
            """
        )

    def _model_servislerini_yukle(self):
        self._durum_yaz("Model servisleri yükleniyor...")
        try:
            from utils.predictor import TahminServisi
            from utils.gradcam import GradCamServisi

            self.tahmin_servisi = TahminServisi()
            self.gradcam_servisi = GradCamServisi(self.tahmin_servisi.model)
            self._olasilik_alanlarini_hazirla()
            self._model_bilgilerini_doldur()
            self._performans_gorsellerini_yukle()
            self._durum_yaz("Sistem hazır. Görsel seçimi sonrası analiz otomatik çalışır.")
        except Exception as hata:  # noqa: BLE001
            self._durum_yaz("Model servisleri yüklenemedi.")
            QMessageBox.critical(
                self,
                "Başlatma Hatası",
                "Uygulama model bileşenlerini yükleyemedi.\n\n"
                "Lütfen eğitim çıktılarının mevcut olduğunu doğrulayın.\n\n"
                f"Ayrıntı: {hata}",
            )

    # ------------------------------------------------------------------
    # Veri setinden hizli secim
    # ------------------------------------------------------------------
    def _dataset_hizli_secim_hazirla(self):
        """dataset/train ve dataset/test klasorlerini analiz eder."""
        self.dataset_haritasi = {}
        self.dataset_duz_liste = {}
        self.dataset_tum_ornekler = []
        for bolum in ("train", "test"):
            bolum_yolu = config.VERI_SETI_DIZIN / bolum
            self.dataset_haritasi[bolum] = {}
            self.dataset_duz_liste[bolum] = []
            if not bolum_yolu.exists():
                continue
            siniflar = sorted([p for p in bolum_yolu.iterdir() if p.is_dir()])
            for sinif_klasor in siniflar:
                dosyalar = sorted(
                    [
                        p
                        for p in sinif_klasor.iterdir()
                        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                    ]
                )
                self.dataset_haritasi[bolum][sinif_klasor.name] = dosyalar
                for p in dosyalar:
                    self.dataset_duz_liste[bolum].append((sinif_klasor.name, p))
                    self.dataset_tum_ornekler.append((bolum, sinif_klasor.name, p))

        self._durum_yaz("Sistem hazır. Veri Setinden Yükle ile rastgele örnek alabilirsiniz.")

    def _hizli_bolum_degisti(self):
        bolum = self.cmb_bolum.currentText()
        siniflar = sorted(self.dataset_haritasi.get(bolum, {}).keys())
        self._hizli_secenek_guncelleniyor = True
        self.cmb_sinif.blockSignals(True)
        self.cmb_sinif.clear()
        self.cmb_sinif.addItems(siniflar)
        self.cmb_sinif.blockSignals(False)
        self._hizli_secenek_guncelleniyor = False
        self._hizli_sinif_degisti()

    def _hizli_sinif_degisti(self):
        bolum = self.cmb_bolum.currentText()
        sinif = self.cmb_sinif.currentText()
        dosyalar = self.dataset_haritasi.get(bolum, {}).get(sinif, [])
        self._hizli_secenek_guncelleniyor = True
        self.cmb_dosya.blockSignals(True)
        self.cmb_dosya.clear()
        self.cmb_dosya.addItems([p.name for p in dosyalar])
        self.cmb_dosya.blockSignals(False)
        self._hizli_secenek_guncelleniyor = False
        self._hizli_dosya_degisti()

    def _hizli_dosya_degisti(self):
        """Kullanici secim degistirdiginde otomatik analiz akisina hazirlar."""
        if self._hizli_secenek_guncelleniyor:
            return
        if self.cmb_dosya.currentText():
            self._durum_yaz("Hızlı seçim hazır. Veri Setinden Yükle ile rastgele örnek alabilirsiniz.")

    def _hizli_secim_uygula(self):
        adaylar = self.dataset_tum_ornekler
        if not adaylar:
            return
        bolum, sinif, yol = random.choice(adaylar)
        dosya = yol.name

        self._durum_yaz(
            f"Rastgele örnek seçildi: {bolum}/{sinif}/{dosya} - analiz başlatılıyor..."
        )
        self._gorseli_yukle_ve_analiz_et(str(yol))

    # ------------------------------------------------------------------
    # Eylemler
    # ------------------------------------------------------------------
    def _gorsel_sec(self):
        dosya, _ = QFileDialog.getOpenFileName(
            self,
            "Görsel Seç",
            str(config.VERI_SETI_DIZIN),
            "Görseller (*.png *.jpg *.jpeg)",
        )
        if not dosya:
            return
        self._gorseli_yukle_ve_analiz_et(dosya)

    def _gorseli_yukle_ve_analiz_et(self, goruntu_yolu: str):
        """Yeni goruntu secildiginde tum analiz akislarini otomatik calistirir."""
        self.secili_goruntu_yolu = goruntu_yolu
        self.lbl_onizleme.setPixmap(
            yola_gore_pixmap_olustur(goruntu_yolu, self.lbl_onizleme.size())
        )
        self.lbl_dosya.setText(f"Dosya: {dosya_adini_kisa_goster(goruntu_yolu)}")
        self._sonuc_alanlarini_temizle()
        self._analiz_calistir()

    def _analiz_calistir(self):
        if not self.secili_goruntu_yolu:
            return
        if self.tahmin_servisi is None:
            QMessageBox.warning(self, "Uyarı", "Model servisi hazır değil.")
            return

        self._durum_yaz("Analiz çalışıyor...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            sonuc = self.tahmin_servisi.tahmin_yap(self.secili_goruntu_yolu)
            self._tahmin_gorunumunu_guncelle(sonuc)
            self._gradcam_gorunumunu_guncelle(sonuc)
            self._durum_yaz("Analiz tamamlandı.")
        except Exception as hata:  # noqa: BLE001
            QMessageBox.critical(self, "Analiz Hatası", f"İşlem başarısız:\n{hata}")
            self._durum_yaz("Analiz sırasında hata oluştu.")
        finally:
            QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------
    # Guncelleme yardimcilari
    # ------------------------------------------------------------------
    def _olasilik_alanlarini_hazirla(self):
        while self.olasilik_grid.count():
            item = self.olasilik_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.prog_cubuklari.clear()
        for satir, model_sinif in enumerate(self.tahmin_servisi.sinif_isimleri):
            gosterim = sinif_adini_kullanici_dostu_yap(model_sinif)
            lbl = QLabel(gosterim)
            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(0)
            pbar.setFormat("%p%")
            pbar.setMinimumWidth(120)
            pbar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.olasilik_grid.addWidget(lbl, satir, 0)
            self.olasilik_grid.addWidget(pbar, satir, 1)
            self.prog_cubuklari[model_sinif] = pbar

    def _tahmin_gorunumunu_guncelle(self, sonuc: dict):
        self.lbl_tahmin.setText(sonuc["sinif_gosterim"])
        sinirda = sonuc.get("sinirda_mi", False)
        if sonuc["bilinen_mi"]:
            guven_metni = f"Güven Skoru: %{sonuc['guven'] * 100:.2f}"
            if sonuc.get("kalibrasyon_t"):
                guven_metni += (
                    f"  (kalibre: %{sonuc['kalibre_guven'] * 100:.2f})"
                )
            if sinirda:
                guven_metni += "  — düşük güvenilirlik, dikkatle yorumlayın"
            self.lbl_guven.setText(guven_metni)
        else:
            self.lbl_guven.setText(
                "Güven Skoru: Geçersiz — görüntü eğitim dağılımının dışında"
            )

        en_yakin = sinif_adini_kullanici_dostu_yap(sonuc["en_yakin_sinif"])
        if not sonuc["bilinen_mi"]:
            self.lbl_meyve_turu.setText(
                f"Meyve Türü: Bilinmiyor (en çok benzediği: {en_yakin})"
            )
            self.lbl_dagilim.setText(
                "Dağılım Durumu: Bilinmeyen Tür — bu görüntü eğitim dağılımının dışında"
            )
        elif sinirda:
            self.lbl_meyve_turu.setText(f"Meyve Türü: {sonuc['meyve_turu']}")
            self.lbl_dagilim.setText(
                "Dağılım Durumu: Sınırda — tahmin gösteriliyor, "
                "görüntü eğitim dağılımının kenarında"
            )
        else:
            self.lbl_meyve_turu.setText(f"Meyve Türü: {sonuc['meyve_turu']}")
            self.lbl_dagilim.setText(f"Dağılım Durumu: {sonuc['dagilim_durumu']}")
        self.lbl_genel_durum.setText(f"Kalite Durumu: {sonuc['genel_durum']}")
        ham = sinif_adini_kullanici_dostu_yap(sonuc["model_sinif"])
        if not sonuc["bilinen_mi"]:
            self.lbl_ham_sinif.setText(
                f"Model önerisi (reddedildi): {ham} "
                f"(%{sonuc['guven'] * 100:.1f})"
            )
        else:
            self.lbl_ham_sinif.setText(f"Tahmin Sınıfı: {ham}")

        m = sonuc["ood_metrikleri"]
        self.lbl_dagilim_metrik.setText(
            "OOD Skoru: "
            f"{m['ood_guven_puani']:.2f}  |  Entropy: {m['entropy']:.2f}  |  "
            f"Mesafe: {m['embedding_distance']:.2f}\n"
            f"Karar Kuralı: {sonuc['tetikleyen_kural']}"
        )

        if sonuc["genel_durum"] == "TAZE":
            self.lbl_genel_durum.setStyleSheet(
                "color: #2f7d47; font-weight:700; background:#e7f6ea; border-radius:6px; padding:4px;"
            )
        elif sonuc["genel_durum"] == "ÇÜRÜK":
            self.lbl_genel_durum.setStyleSheet(
                "color: #7a1e2d; font-weight:700; background:#f6e8eb; border-radius:6px; padding:4px;"
            )
        else:
            self.lbl_genel_durum.setStyleSheet(
                "color: #5a4a3f; font-weight:700; background:#f2ece6; border-radius:6px; padding:4px;"
            )

        if not sonuc["bilinen_mi"]:
            self.lbl_dagilim.setStyleSheet(
                "color: #7a1e2d; font-weight:700; background:#f6e8eb; border-radius:6px; padding:4px;"
            )
        elif sonuc.get("sinirda_mi", False):
            self.lbl_dagilim.setStyleSheet(
                "color: #8a5a14; font-weight:700; background:#faf0dc; border-radius:6px; padding:4px;"
            )
        else:
            self.lbl_dagilim.setStyleSheet(
                "color: #2f7d47; font-weight:700; background:#e7f6ea; border-radius:6px; padding:4px;"
            )

        for sinif, pbar in self.prog_cubuklari.items():
            pbar.setValue(round(sonuc["olasiliklar"].get(sinif, 0) * 100))

    def _gradcam_gorunumunu_guncelle(self, sonuc: dict):
        analiz = self.gradcam_servisi.analiz_yap(self.secili_goruntu_yolu)
        self.lbl_grad_orijinal.setPixmap(
            diziye_gore_pixmap_olustur(analiz["orijinal"], self.lbl_grad_orijinal.size())
        )
        self.lbl_grad_overlay.setPixmap(
            diziye_gore_pixmap_olustur(analiz["overlay"], self.lbl_grad_overlay.size())
        )
        hedef = sinif_adini_kullanici_dostu_yap(sonuc["model_sinif"])
        self.lbl_grad_tahmin.setText(
            f"Hedef sınıf (Grad-CAM): {hedef}  —  "
            f"Güven: %{sonuc['guven'] * 100:.1f}"
        )
        self.lbl_katman.setText(
            f"Hedef katman: {analiz['katman']} "
            "(MobileNetV2 son konvolüsyon bloğu)"
        )
        if sonuc.get("bilinen_mi", True) and not sonuc.get("sinirda_mi", False):
            self.lbl_grad_bilgi.setText(
                "Isı haritasındaki sıcak renkler, modelin kararında daha etkili bölgeleri gösterir."
            )
        elif sonuc.get("sinirda_mi", False):
            self.lbl_grad_bilgi.setText(
                "Bu görüntü eğitim dağılımının kenarında (sınırda) işaretlendi. "
                "Isı haritası modelin odağını gösterir; tahmin doğru olabilir "
                "ancak düşük güvenle yorumlanmalıdır."
            )
        else:
            en_yakin = sinif_adini_kullanici_dostu_yap(
                sonuc.get("en_yakin_sinif", "")
            )
            self.lbl_grad_bilgi.setText(
                "Bu görüntü eğitim dağılımının dışında işaretlendi "
                f"(embedding uzayında en çok benzediği sınıf: {en_yakin}). "
                "Grad-CAM gösterimi üretildi; ancak sınıf kararı geçersiz sayılmalıdır."
            )

    def _model_bilgilerini_doldur(self):
        while self.model_bilgi_layout.count():
            item = self.model_bilgi_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for baslik, alanlar in self.tahmin_servisi.model_bilgi_bolumleri():
            if not alanlar:
                continue
            grup = QGroupBox(baslik)
            grid = QGridLayout(grup)
            grid.setHorizontalSpacing(16)
            grid.setVerticalSpacing(6)
            grid.setColumnStretch(1, 1)
            for satir, (anahtar, deger) in enumerate(alanlar.items()):
                lbl_k = QLabel(f"{anahtar}:")
                lbl_k.setStyleSheet("font-weight: 700; color:#4a3029;")
                lbl_v = QLabel(str(deger))
                lbl_v.setWordWrap(True)
                lbl_v.setStyleSheet("color:#3f3a35;")
                grid.addWidget(lbl_k, satir, 0, alignment=Qt.AlignTop)
                grid.addWidget(lbl_v, satir, 1, alignment=Qt.AlignTop)
            self.model_bilgi_layout.addWidget(grup)
        self.model_bilgi_layout.addStretch(1)

    def _performans_ozetini_doldur(self):
        metrik_yolu = config.RAPOR_DIZIN / "metrikler.json"
        if not metrik_yolu.exists():
            self.lbl_perf_ozet.setText(
                "Test metrikleri henüz üretilmedi. `python evaluate.py` çalıştırın."
            )
            return
        import json

        metrik = json.loads(metrik_yolu.read_text(encoding="utf-8"))
        acc = metrik.get("accuracy", 0) * 100
        f1 = metrik.get("f1_makro", 0)
        self.lbl_perf_ozet.setText(
            f"Genel test doğruluğu: %{acc:.2f}  |  Macro F1: {f1:.3f}  |  "
            f"5.399 test görseli, 16 sınıf.\n"
            "Güçlü sınıflar: elma, muz, portakal, çilek. "
            "İyileştirme adayları: çürük üzüm (recall ~%43), çürük hünnap (~%69), "
            "çürük nar (~%72)."
        )

    def _performans_gorsellerini_yukle(self):
        self._performans_ozetini_doldur()
        self._performans_yollari = {
            self.lbl_acc_img: config.CIKTI_DIZIN / "accuracy.png",
            self.lbl_loss_img: config.CIKTI_DIZIN / "loss.png",
            self.lbl_cm_img: config.CIKTI_DIZIN / "confusion_matrix.png",
        }
        for hedef, yol in self._performans_yollari.items():
            self._tek_gorsel_yukle(hedef, yol)

    def _tek_gorsel_yukle(self, hedef_etiket: TiklanabilirGorselEtiketi, yol: Path):
        if yol.exists():
            hedef_etiket.buyuk_goruntu_yolu = str(yol)
            hedef_boyut = hedef_etiket.size()
            if hedef_boyut.width() < 500 or hedef_boyut.height() < 180:
                hedef_boyut = QSize(1150, 320)
            hedef_etiket.setPixmap(
                yola_gore_pixmap_olustur(str(yol), hedef_boyut)
            )
            return
        hedef_etiket.setText(f"Görsel bulunamadı:\n{yol.name}")

    def resizeEvent(self, event):  # noqa: N802
        """Pencere boyutu degistiginde gorselleri yeniden olcekler."""
        super().resizeEvent(event)
        if self.secili_goruntu_yolu:
            self.lbl_onizleme.setPixmap(
                yola_gore_pixmap_olustur(
                    self.secili_goruntu_yolu, self.lbl_onizleme.size()
                )
            )
        if self._performans_yollari:
            for hedef, yol in self._performans_yollari.items():
                self._tek_gorsel_yukle(hedef, yol)

    def _gorseli_buyuk_ac(self, goruntu_yolu: str):
        """Performans gorsellerini buyuk pencerede acar."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Görsel Önizleme")
        dialog.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        dialog.resize(1000, 720)
        yer = QVBoxLayout(dialog)

        ust = QHBoxLayout()
        ust.addStretch(1)
        btn_detay = QToolButton()
        btn_detay.setText("?")
        btn_detay.setToolTip("Grafik yolu ve teknik açıklamayı göster")
        btn_detay.clicked.connect(
            lambda: self._grafik_detay_penceresi_ac(goruntu_yolu)
        )
        ust.addWidget(btn_detay)
        yer.addLayout(ust)

        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setPixmap(yola_gore_pixmap_olustur(goruntu_yolu, dialog.size()))
        yer.addWidget(lbl)
        dialog.exec_()

    def _grafik_detay_penceresi_ac(self, goruntu_yolu: str):
        """Grafik dosya yolunu ve teknik yorumunu bilgi kutusunda gosterir."""
        ad = Path(goruntu_yolu).name.lower()
        if "accuracy" in ad:
            teknik = (
                "Accuracy grafiği epoch bazında doğru sınıflandırma oranını izler.\n"
                "- Mavi çizgi eğitim doğruluğu, turuncu çizgi doğrulama doğruluğudur.\n"
                "- Eğrilerin birlikte artması modelin öğrenmesinin dengeli olduğunu gösterir.\n"
                "- Çizgiler çok ayrışırsa overfitting riski artar."
            )
        elif "loss" in ad:
            teknik = (
                "Loss grafiği model hatasının eğitim boyunca nasıl değiştiğini gösterir.\n"
                "- Eğitim ve doğrulama loss değerlerinin düşmesi beklenir.\n"
                "- Doğrulama loss yükselirken eğitim loss düşüyorsa ezberleme başlayabilir.\n"
                "- İnce ayar başlangıcındaki geçici sıçramalar normal kabul edilebilir."
            )
        elif "confusion" in ad:
            teknik = (
                "Confusion Matrix sınıf bazında hata dağılımını gösterir.\n"
                "- Köşegen hücreler doğru tahminleri temsil eder.\n"
                "- Köşegen dışı hücreler hangi sınıfların karıştırıldığını gösterir.\n"
                "- Taze/çürük ayrımındaki kritik hataları bu tablo üzerinden izleriz."
            )
        else:
            teknik = "Bu görsel için kayıtlı teknik açıklama bulunamadı."

        QMessageBox.information(
            self,
            "Grafik Teknik Bilgi",
            f"Grafik Dosyası:\n{goruntu_yolu}\n\nTeknik Açıklama:\n{teknik}",
        )

    def _sonuc_alanlarini_temizle(self):
        self.lbl_tahmin.setText("-")
        self.lbl_guven.setText("Güven Skoru: -")
        self.lbl_meyve_turu.setText("Meyve Türü: -")
        self.lbl_genel_durum.setText("Kalite Durumu: -")
        self.lbl_ham_sinif.setText("Tahmin Sınıfı: -")
        self.lbl_dagilim.setText("Dağılım Durumu: -")
        self.lbl_dagilim_metrik.setText("OOD Skoru / Entropy / Mesafe: -")
        self.lbl_genel_durum.setStyleSheet("")
        self.lbl_dagilim.setStyleSheet("")

        for pbar in self.prog_cubuklari.values():
            pbar.setValue(0)

        self.lbl_grad_orijinal.clear()
        self.lbl_grad_orijinal.setText("Görsel bekleniyor")
        self.lbl_grad_overlay.clear()
        self.lbl_grad_overlay.setText("Analiz bekleniyor")
        self.lbl_grad_tahmin.setText("Hedef sınıf: Analiz bekleniyor")
        self.lbl_katman.setText("Hedef katman: -")

    def _durum_yaz(self, mesaj: str):
        self.lbl_durum.setText(mesaj)
