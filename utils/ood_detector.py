"""OOD (Out-of-Distribution) / Bilinmeyen Meyve algilayici.

Bu modul, modeli bozmadan ve yeniden buyuk capli egitim gerektirmeden
asagidaki 3 metriği birlikte kullanir:
1) Maksimum olasilik
2) Entropy (belirsizlik)
3) Embedding uzakligi (sinif merkezlerine cosine distance)

Sonuc:
- Bilinen Tür
- Bilinmeyen Tür
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import tensorflow as tf

from src import config


class OODAlgilayici:
    """Bilinmeyen meyve kararini hibrit metriklerle uretir."""

    def __init__(self, model, sinif_isimleri: List[str]):
        self.model = model
        self.sinif_isimleri = sinif_isimleri
        self.embedding_model = self._embedding_model_olustur()
        self.istatistikler = self._istatistikleri_yukle_veya_uret()

    def _embedding_model_olustur(self):
        """Siniflandirma oncesi embedding vektorunu veren modeli hazirlar."""
        aday_katmanlar = ["dropout", "global_havuz"]
        for katman_adi in aday_katmanlar:
            try:
                cikis = self.model.get_layer(katman_adi).output
                return tf.keras.Model(self.model.input, cikis)
            except ValueError:
                continue
        # Yedek: softmax oncesi son katman cikisini hedefle.
        return tf.keras.Model(self.model.input, self.model.layers[-2].output)

    def _istatistikleri_yukle_veya_uret(self) -> Dict[str, Any]:
        """Uyumlu cache varsa yukler, yoksa train setinden yeniden hesaplar."""
        dosya = config.OOD_ISTATISTIK_DOSYA
        if dosya.exists():
            icerik = json.loads(dosya.read_text(encoding="utf-8"))
            if icerik.get("sinif_isimleri") == self.sinif_isimleri:
                return self._jsondan_numpyye(icerik)

        ist = self._train_verisinden_istatistik_hesapla()
        config.klasorleri_hazirla()
        dosya.write_text(
            json.dumps(self._numpydan_jsone(ist), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return ist

    def _train_verisinden_istatistik_hesapla(self) -> Dict[str, Any]:
        """Train setinden embedding dagilimini ve karar esiklerini cikarir."""
        sinif_bazli_embedding = {}
        tum_embedding = []
        tum_max_prob = []
        tum_entropy = []

        for sinif in self.sinif_isimleri:
            klasor = config.EGITIM_DIZIN / sinif
            dosyalar = sorted(
                [
                    p
                    for p in klasor.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]
            )[: config.OOD_SINIF_BASI_AZAMI_ORNEK]
            if not dosyalar:
                continue

            girdi = self._batch_onisle(dosyalar)
            olasilik = self.model.predict(girdi, verbose=0)
            embedding = self.embedding_model.predict(girdi, verbose=0)

            # Entropy: normalize edilmis (0-1 arasi)
            entropy = self._normalize_entropy(olasilik)
            max_prob = np.max(olasilik, axis=1)

            sinif_bazli_embedding[sinif] = embedding
            tum_embedding.append(embedding)
            tum_max_prob.append(max_prob)
            tum_entropy.append(entropy)

        if not tum_embedding:
            raise RuntimeError("OOD istatistikleri icin train embedding cikarilamadi.")

        # Sinif merkezleri
        sinif_merkezleri = {
            sinif: np.mean(emb, axis=0) for sinif, emb in sinif_bazli_embedding.items()
        }

        # Bilinen dagilimda nearest centroid distance dagilimi
        tum_embedding_np = np.concatenate(tum_embedding, axis=0)
        nearest_dist = np.array(
            [
                self._en_yakin_merkez_uzakligi(v, sinif_merkezleri)
                for v in tum_embedding_np
            ]
        )

        tum_max_prob_np = np.concatenate(tum_max_prob, axis=0)
        tum_entropy_np = np.concatenate(tum_entropy, axis=0)

        # Esikler: bilinen dagilimdan istatistiksel percentile tabanli.
        esikler = {
            "max_prob_alt_esik": float(np.percentile(tum_max_prob_np, 10)),
            "entropy_ust_esik": float(np.percentile(tum_entropy_np, 90)),
            "distance_ust_esik": float(np.percentile(nearest_dist, 95)),
        }

        return {
            "surum": 1,
            "olusturma_zamani": datetime.now().isoformat(timespec="seconds"),
            "sinif_isimleri": self.sinif_isimleri,
            "sinif_merkezleri": sinif_merkezleri,
            "esikler": esikler,
            "ornek_sayisi": int(len(tum_embedding_np)),
            "ornekleme_azami": int(config.OOD_SINIF_BASI_AZAMI_ORNEK),
        }

    def degerlendir(self, model_girisi: np.ndarray, olasiliklar: np.ndarray) -> Dict[str, Any]:
        """Tek goruntu icin uc kademeli OOD kararini dondurur.

        Mesafe-oncelikli uc kademeli karar:
        1) mesafe > T_yuksek -> Bilinmeyen (softmax guveni bu karari ezemez)
        2) mesafe < T_dusuk  -> Bilinen
        3) gri bolge         -> softmax sinifi ile en yakin merkez sinifi
           uyusuyorsa ve guven kosullari saglaniyorsa "Sinirda" (tahmin dusuk
           guven uyarisiyla gosterilir); aksi halde Bilinmeyen.

        Gerekce: embedding uzayinda gercek-ama-alisilmadik meyveler (sari
        burusuk elma) ile yabanci meyveler (murdum erigi) ic ice gectiginden
        tek esikle ayrilamaz; uc kademeli karar sinirda vakalarda tahmini
        gizlemek yerine acik belirsizlik uyarisi verir. Detaylar:
        reports/ood_teshis_raporu.md
        """
        embedding = self.embedding_model.predict(model_girisi, verbose=0)[0]
        max_prob = float(np.max(olasiliklar))
        entropy = float(self._normalize_entropy(olasiliklar[np.newaxis, :])[0])
        merkezler = self.istatistikler["sinif_merkezleri"]
        uzaklik = float(self._en_yakin_merkez_uzakligi(embedding, merkezler))
        en_yakin_sinif = self._en_yakin_sinif(embedding, merkezler)
        softmax_sinif = self.sinif_isimleri[int(np.argmax(olasiliklar))]
        tahmin_merkez_mesafe = float(
            self._sinifa_mesafe(embedding, softmax_sinif, merkezler)
        )

        esik_prob = self.istatistikler["esikler"]["max_prob_alt_esik"]
        esik_ent = self.istatistikler["esikler"]["entropy_ust_esik"]
        esik_dist = self.istatistikler["esikler"]["distance_ust_esik"]

        # Metrikleri 0-1 arasi puanlayip birlestir (UI'daki bilgi karti icin).
        prob_puani = self._birim_puan(max_prob, esik_prob, ters=False)
        ent_puani = self._birim_puan(entropy, esik_ent, ters=True)
        dist_puani = self._birim_puan(uzaklik, esik_dist, ters=True)
        birlesik_puan = 0.40 * prob_puani + 0.25 * ent_puani + 0.35 * dist_puani

        t_yuksek = config.OOD_MESAFE_T_YUKSEK
        t_dusuk = config.OOD_MESAFE_T_DUSUK
        durum, tetikleyen_kural = self.hiyerarsik_karar(
            uzaklik,
            max_prob,
            entropy,
            softmax_sinif=softmax_sinif,
            en_yakin_sinif=en_yakin_sinif,
            tahmin_merkez_mesafe=tahmin_merkez_mesafe,
        )

        durum_metni = {
            "bilinen": "Bilinen Tür",
            "sinirda": "Sınırda (Düşük Güven)",
            "bilinmeyen": "Bilinmeyen Tür",
        }[durum]

        return {
            "dagilim_durumu": durum_metni,
            "durum_kodu": durum,
            # Sinirda durumda da tahmin gosterilir; yalnizca "bilinmeyen"
            # durumunda sinif bilgisi gizlenir.
            "bilinen_mi": durum != "bilinmeyen",
            "sinirda_mi": durum == "sinirda",
            "max_prob": max_prob,
            "entropy": entropy,
            "embedding_distance": uzaklik,
            "en_yakin_sinif": en_yakin_sinif,
            "tahmin_merkez_mesafe": tahmin_merkez_mesafe,
            "ood_guven_puani": float(birlesik_puan),
            "tetikleyen_kural": tetikleyen_kural,
            "esikler": {
                "max_prob_alt_esik": esik_prob,
                "entropy_ust_esik": esik_ent,
                "distance_ust_esik": esik_dist,
            },
            "hiyerarsi_esikleri": {
                "mesafe_t_yuksek": t_yuksek,
                "mesafe_t_dusuk": t_dusuk,
                "gri_maxprob_esik": config.OOD_GRI_MAXPROB_ESIK,
                "gri_entropy_esik": config.OOD_GRI_ENTROPY_ESIK,
            },
            "karar_ozeti": {
                "mesafe_oncelikli_bilinmeyen": bool(uzaklik > t_yuksek),
                "bilinen_bolge": bool(uzaklik < t_dusuk),
                "gri_bolge": bool(t_dusuk <= uzaklik <= t_yuksek),
                "softmax_merkez_uyumu": bool(softmax_sinif == en_yakin_sinif),
                "tetikleyen_kural": tetikleyen_kural,
            },
        }

    @staticmethod
    def _meyve_ailesi(sinif: str | None) -> str:
        """Sinif adindan meyve turu ailesini cikarir (elma/muz/portakal)."""
        if not sinif:
            return ""
        temiz = sinif.lower()
        if "apple" in temiz:
            return "elma"
        if "banana" in temiz:
            return "muz"
        if "orange" in temiz:
            return "portakal"
        return ""

    @staticmethod
    def hiyerarsik_karar(
        uzaklik: float,
        max_prob: float,
        entropy: float,
        softmax_sinif: str = None,
        en_yakin_sinif: str = None,
        tahmin_merkez_mesafe: float = None,
        t_yuksek: float = None,
        t_dusuk: float = None,
        gri_maxprob: float = None,
        gri_entropy: float = None,
        gri_yuksek_guven: float = None,
    ):
        """Mesafe-oncelikli UC KADEMELI karari saf fonksiyon olarak uygular.

        Oncelik: (1) kesin bilinmeyen mesafe, (2) bilinen bolge, (3) gri
        bolgede model tahmini veya yuksek softmax guveni ile sinirda uyari,
        (4) dusuk guvenli / celiskili yabanci girdiler bilinmeyen.

        Donen deger: (durum, tetikleyen_kural)
        """
        t_yuksek = config.OOD_MESAFE_T_YUKSEK if t_yuksek is None else t_yuksek
        t_dusuk = config.OOD_MESAFE_T_DUSUK if t_dusuk is None else t_dusuk
        gri_maxprob = config.OOD_GRI_MAXPROB_ESIK if gri_maxprob is None else gri_maxprob
        gri_entropy = config.OOD_GRI_ENTROPY_ESIK if gri_entropy is None else gri_entropy
        gri_yuksek_guven = (
            config.OOD_GRI_YUKSEK_GUVEN if gri_yuksek_guven is None else gri_yuksek_guven
        )

        if uzaklik > t_yuksek:
            return "bilinmeyen", f"Mesafe > T_yüksek ({uzaklik:.3f} > {t_yuksek:.3f})"
        if uzaklik < t_dusuk:
            return "bilinen", f"Bilinen sınıf bölgesinde ({uzaklik:.3f} < {t_dusuk:.3f})"

        uyum = (
            softmax_sinif is None
            or en_yakin_sinif is None
            or softmax_sinif == en_yakin_sinif
        )
        aile_uyum = (
            OODAlgilayici._meyve_ailesi(softmax_sinif)
            and OODAlgilayici._meyve_ailesi(softmax_sinif)
            == OODAlgilayici._meyve_ailesi(en_yakin_sinif)
        )
        guvenli = max_prob >= gri_maxprob and entropy <= gri_entropy

        # Tahmin sinifinin kendi merkezine yakin: gercek ama alisilmadik ornekler
        # (sari elma) icin model tahminini koru.
        if (
            tahmin_merkez_mesafe is not None
            and tahmin_merkez_mesafe < t_dusuk
        ):
            return "sinirda", (
                f"Gri bölge; tahmin sınıfı merkezine yakın "
                f"(d_tahmin={tahmin_merkez_mesafe:.3f})"
            )

        if uyum and guvenli:
            return "sinirda", (
                f"Gri bölge + softmax-merkez uyumu (mesafe {uzaklik:.3f}, "
                f"maxprob {max_prob:.2f})"
            )

        # Yuksek softmax: merkez celiskisi olsa bile model tahminini goster
        # (orn. %95 rotten apple + en yakin merkez portakal -> sinirda).
        if max_prob >= gri_yuksek_guven:
            return "sinirda", (
                f"Gri bölge + yüksek model güveni (maxprob {max_prob:.2f})"
            )

        if aile_uyum and max_prob >= gri_maxprob:
            return "sinirda", (
                f"Gri bölge + aynı meyve türü (softmax: {softmax_sinif}, "
                f"merkez: {en_yakin_sinif})"
            )

        if not uyum:
            return "bilinmeyen", (
                f"Gri bölge + düşük güvenli sinyal uyuşmazlığı "
                f"(softmax: {softmax_sinif}, merkez: {en_yakin_sinif}, "
                f"maxprob {max_prob:.2f})"
            )
        return "bilinmeyen", (
            f"Gri bölge + yetersiz güven (maxprob {max_prob:.2f}, entropy {entropy:.2f})"
        )

    def _sinifa_mesafe(
        self,
        vektor: np.ndarray,
        sinif: str,
        merkezler: Dict[str, np.ndarray],
    ) -> float:
        """Verilen sinif merkezine cosine uzakligi."""
        if sinif not in merkezler:
            return 1.0
        return self._cosine_distance(vektor, merkezler[sinif])

    @staticmethod
    def _batch_onisle(dosyalar: List[Path]) -> np.ndarray:
        """Goruntuleri batched tensor haline getirir."""
        img_dizileri = []
        for p in dosyalar:
            img = tf.keras.utils.load_img(p, target_size=config.GORUNTU_BOYUTU)
            arr = tf.keras.utils.img_to_array(img)
            img_dizileri.append(arr)
        x = np.array(img_dizileri, dtype=np.float32)
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)

    @staticmethod
    def _normalize_entropy(olasiliklar: np.ndarray) -> np.ndarray:
        """Entropy'yi 0-1 araligina normalize eder."""
        eps = 1e-12
        p = np.clip(olasiliklar, eps, 1.0)
        ent = -np.sum(p * np.log(p), axis=1)
        max_ent = np.log(p.shape[1])
        return ent / max_ent

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine uzakligi (0 iyi, 1 kotu)."""
        eps = 1e-12
        a_norm = np.linalg.norm(a) + eps
        b_norm = np.linalg.norm(b) + eps
        cos_sim = float(np.dot(a, b) / (a_norm * b_norm))
        return 1.0 - cos_sim

    def _en_yakin_merkez_uzakligi(self, vektor: np.ndarray, merkezler: Dict[str, np.ndarray]) -> float:
        """Vektore en yakin sinif merkezinin uzakligini bulur."""
        uzakliklar = [self._cosine_distance(vektor, m) for m in merkezler.values()]
        return float(min(uzakliklar))

    def _en_yakin_sinif(self, vektor: np.ndarray, merkezler: Dict[str, np.ndarray]) -> str:
        """Vektore en yakin sinif merkezinin adini dondurur."""
        return min(
            merkezler.items(),
            key=lambda kv: self._cosine_distance(vektor, kv[1]),
        )[0]

    @staticmethod
    def _birim_puan(deger: float, esik: float, ters: bool) -> float:
        """Degeri esige gore 0-1 arasi normalize eder.

        ters=False: buyuk deger daha iyi (max_prob gibi)
        ters=True : kucuk deger daha iyi (entropy, distance gibi)
        """
        eps = 1e-8
        if not ters:
            # esik alti 0'a, 1'e dogru yukari.
            return float(np.clip((deger - esik) / (1.0 - esik + eps), 0.0, 1.0))
        # esik alti iyi, ustu kotu.
        return float(np.clip((esik - deger) / (esik + eps), 0.0, 1.0))

    @staticmethod
    def _numpydan_jsone(ist: Dict[str, Any]) -> Dict[str, Any]:
        """Numpy dizilerini JSON uyumlu listeye cevirir."""
        c = dict(ist)
        c["sinif_merkezleri"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in c["sinif_merkezleri"].items()
        }
        return c

    @staticmethod
    def _jsondan_numpyye(ist: Dict[str, Any]) -> Dict[str, Any]:
        """JSON listelerini numpy diziye geri cevirir."""
        c = dict(ist)
        c["sinif_merkezleri"] = {
            k: np.array(v, dtype=np.float32)
            for k, v in c["sinif_merkezleri"].items()
        }
        return c
