"""Model yukleme ve tahmin servis katmani."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from predict import model_yukle
from src import config
from src.utils import goruntu_yukle_ve_hazirla
from utils.image_utils import (
    genel_durumu_uret,
    meyve_turunu_uret,
    sinif_adini_kullanici_dostu_yap,
)
from utils.ood_detector import OODAlgilayici


def _dataset_ornegi_mi(goruntu_yolu: str) -> bool:
    """Gorsel proje dataset/ klasorunden mi geliyor?"""
    try:
        Path(goruntu_yolu).resolve().relative_to(config.VERI_SETI_DIZIN.resolve())
        return True
    except (ValueError, OSError):
        return False


class TahminServisi:
    """Tek noktadan model yukleme ve tahmin islemlerini yonetir."""

    def __init__(self):
        self.model = None
        self.sinif_isimleri = []
        self.ood_algilayici = None
        self.kalibrasyon_t = None
        self._yukle()

    def _yukle(self):
        """Modeli ve sinif eslemesini bellege alir."""
        self.model = model_yukle()
        if not config.SINIF_INDEKS_DOSYA.exists():
            raise FileNotFoundError(
                "Sinif esleme dosyasi bulunamadi (class_indices.json)."
            )
        esleme = json.loads(config.SINIF_INDEKS_DOSYA.read_text(encoding="utf-8"))
        self.sinif_isimleri = [esleme[str(i)] for i in range(len(esleme))]
        self.ood_algilayici = OODAlgilayici(self.model, self.sinif_isimleri)

        # Temperature scaling parametresi (varsa). Tahmin sinifini degistirmez;
        # yalnizca gosterilen guven yuzdesini kalibre eder.
        if config.KALIBRASYON_DOSYA.exists():
            kalibrasyon = json.loads(
                config.KALIBRASYON_DOSYA.read_text(encoding="utf-8")
            )
            self.kalibrasyon_t = float(kalibrasyon.get("T", 1.0))

    def _kalibre_olasiliklar(self, olasilik_dizisi: np.ndarray) -> np.ndarray:
        """Softmax olasiliklarini T sicakligiyla yeniden olcekler (argmax sabit)."""
        if self.kalibrasyon_t is None or self.kalibrasyon_t == 1.0:
            return olasilik_dizisi
        eps = 1e-12
        log_p = np.log(np.clip(olasilik_dizisi, eps, 1.0)) / self.kalibrasyon_t
        log_p -= log_p.max()
        p = np.exp(log_p)
        return p / p.sum()

    def tahmin_yap(self, goruntu_yolu: str) -> Dict[str, Any]:
        """Verilen goruntu icin detayli tahmin sonuclari dondurur."""
        model_girisi, _ = goruntu_yukle_ve_hazirla(goruntu_yolu)
        olasilik_dizisi = self.model.predict(model_girisi, verbose=0)[0]
        en_iyi_indeks = int(np.argmax(olasilik_dizisi))
        model_sinif = self.sinif_isimleri[en_iyi_indeks]

        olasiliklar = {
            ad: float(olasilik)
            for ad, olasilik in zip(self.sinif_isimleri, olasilik_dizisi)
        }
        olasiliklar = dict(
            sorted(olasiliklar.items(), key=lambda x: x[1], reverse=True)
        )

        ood = self.ood_algilayici.degerlendir(model_girisi, olasilik_dizisi)
        dataset_ornegi = _dataset_ornegi_mi(goruntu_yolu)

        if dataset_ornegi:
            # Veri seti goruntuleri egitim dagiliminin parcasidir; OOD uyarisi
            # model tahminiyle celisir (orn. %100 guven + sinirda uyarisi).
            bilinen = True
            sinirda = False
            dagilim_durumu = "Bilinen Tür (veri seti örneği)"
            tetikleyen_kural = "Veri seti kaynağı — OOD katmanı devre dışı"
        else:
            bilinen = ood["bilinen_mi"]
            sinirda = ood["sinirda_mi"]
            dagilim_durumu = ood["dagilim_durumu"]
            tetikleyen_kural = ood["tetikleyen_kural"]

        kalibre = self._kalibre_olasiliklar(olasilik_dizisi)
        kalibre_guven = float(kalibre[en_iyi_indeks])

        if not bilinen:
            sinif_gosterim = "Bilinmeyen Meyve"
        elif sinirda:
            sinif_gosterim = f"{sinif_adini_kullanici_dostu_yap(model_sinif)} (?)"
        else:
            sinif_gosterim = sinif_adini_kullanici_dostu_yap(model_sinif)

        sonuc = {
            "model_sinif": model_sinif,
            "sinif_gosterim": sinif_gosterim,
            "guven": float(olasilik_dizisi[en_iyi_indeks]),
            "kalibre_guven": kalibre_guven,
            "kalibrasyon_t": self.kalibrasyon_t,
            "genel_durum": (
                genel_durumu_uret(model_sinif) if bilinen else "Belirlenemedi"
            ),
            "meyve_turu": (
                meyve_turunu_uret(model_sinif) if bilinen else "Bilinmiyor"
            ),
            "dagilim_durumu": dagilim_durumu,
            "durum_kodu": "bilinen" if dataset_ornegi else ood["durum_kodu"],
            "bilinen_mi": bilinen,
            "sinirda_mi": sinirda,
            "dataset_ornegi": dataset_ornegi,
            "en_yakin_sinif": ood["en_yakin_sinif"],
            "tetikleyen_kural": tetikleyen_kural,
            "ood_metrikleri": {
                "max_prob": ood["max_prob"],
                "entropy": ood["entropy"],
                "embedding_distance": ood["embedding_distance"],
                "ood_guven_puani": ood["ood_guven_puani"],
            },
            "olasiliklar": olasiliklar,
        }
        return sonuc

    @staticmethod
    def _goruntu_sayisi(dizin: Path) -> int:
        if not dizin.exists():
            return 0
        return sum(
            1
            for sinif in dizin.iterdir()
            if sinif.is_dir()
            for p in sinif.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

    def model_bilgi_bolumleri(self) -> list[tuple[str, Dict[str, str]]]:
        """Model bilgilerini mantiksal bolumlere ayirarak dondurur."""
        meyve_sayisi = len({meyve_turunu_uret(s) for s in self.sinif_isimleri})
        train_adet = self._goruntu_sayisi(config.EGITIM_DIZIN)
        test_adet = self._goruntu_sayisi(config.TEST_DIZIN)

        genel = {
            "Mimari": "MobileNetV2 (ImageNet ön-eğitimli)",
            "Yaklaşım": "İki aşamalı transfer learning + fine-tuning",
            "Sınıf sayısı": f"{len(self.sinif_isimleri)} ({meyve_sayisi} meyve × taze/çürük)",
            "Görüntü boyutu": f"{config.GORUNTU_BOYUTU[0]}×{config.GORUNTU_BOYUTU[1]} px",
            "Batch boyutu": str(config.YIGIN_BOYUTU),
        }

        veri = {
            "Eğitim görseli": f"{train_adet:,}",
            "Test görseli": f"{test_adet:,}",
            "Kaynak": "Orijinal veri seti + Mendeley Augmented Image (16 sınıf)",
            "Meyveler": "Elma, muz, portakal, üzüm, guava, hünnap, nar, çilek",
        }

        egitim = {
            "Planlanan epoch": (
                f"Aşama 1: {config.ASAMA1_EPOCH} + Aşama 2: {config.ASAMA2_EPOCH}"
            ),
            "Early stopping": f"val_loss, sabır={config.ERKEN_DURDURMA_SABIR}",
        }
        if config.EGITIM_GECMIS_DOSYA.exists():
            gecmis = json.loads(config.EGITIM_GECMIS_DOSYA.read_text(encoding="utf-8"))
            val_acc = gecmis.get("val_accuracy", [])
            epoch_sayisi = len(gecmis.get("accuracy", []))
            egitim["Tamamlanan epoch"] = str(epoch_sayisi)
            if val_acc:
                en_iyi_idx = int(np.argmax(val_acc))
                egitim["En iyi epoch"] = str(en_iyi_idx + 1)
                egitim["En iyi val accuracy"] = f"%{val_acc[en_iyi_idx] * 100:.2f}"
            asama1 = config.ASAMA1_EPOCH
            if epoch_sayisi > asama1:
                egitim["Fine-tuning"] = (
                    f"Aşama 2'de {epoch_sayisi - asama1} epoch tamamlandı"
                )
            elif epoch_sayisi <= asama1:
                egitim["Fine-tuning"] = (
                    f"Aşama 1'de early stopping (epoch {epoch_sayisi})"
                )

        test = {}
        metrik_yolu = config.RAPOR_DIZIN / "metrikler.json"
        if metrik_yolu.exists():
            metrik = json.loads(metrik_yolu.read_text(encoding="utf-8"))
            if "accuracy" in metrik:
                test["Test doğruluğu"] = f"%{metrik['accuracy'] * 100:.2f}"
            if "f1_makro" in metrik:
                test["Macro F1"] = f"{metrik['f1_makro']:.3f}"
            if "f1_agirlikli" in metrik:
                test["Weighted F1"] = f"{metrik['f1_agirlikli']:.3f}"
            test["Değerlendirme seti"] = f"{test_adet:,} görsel (train/test ayrı)"

        ood = {
            "Karar modeli": "Üç kademeli: Bilinen / Sınırda / Bilinmeyen",
            "Mesafe T_yüksek": f"{config.OOD_MESAFE_T_YUKSEK:.3f}",
            "Mesafe T_düşük": f"{config.OOD_MESAFE_T_DUSUK:.3f}",
            "Gri bölge max prob": f"{config.OOD_GRI_MAXPROB_ESIK:.2f}",
            "Gri bölge entropy": f"{config.OOD_GRI_ENTROPY_ESIK:.2f}",
        }
        if self.kalibrasyon_t is not None:
            ood["Güven kalibrasyonu"] = f"Temperature scaling (T={self.kalibrasyon_t:.2f})"

        return [
            ("Genel Bakış", genel),
            ("Veri Seti", veri),
            ("Eğitim Süreci", egitim),
            ("Test Performansı", test),
            ("OOD ve Güven", ood),
        ]

    def model_bilgi_ozeti(self) -> Dict[str, str]:
        """Geriye uyumluluk icin duz sozluk ozeti."""
        ozet: Dict[str, str] = {}
        for _, bolum in self.model_bilgi_bolumleri():
            ozet.update(bolum)
        return ozet

    @staticmethod
    def olasilik_metnini_duzenle(sinif_adi: str, olasilik: float) -> str:
        """Model sinif adini kullanici dostu metne cevirir."""
        return f"{sinif_adini_kullanici_dostu_yap(sinif_adi)}  %{olasilik * 100:.2f}"
