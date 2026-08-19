"""Arayuz icin Grad-CAM servis katmani."""

from __future__ import annotations

import numpy as np

from gradcam import grad_model_olustur, isi_haritasi_uret, overlay_olustur
from src.utils import goruntu_yukle_ve_hazirla


class GradCamServisi:
    """Yuklenmis model uzerinden Grad-CAM analizi uretir."""

    def __init__(self, model):
        self.model = model
        self.grad_model, self.son_katman_adi = grad_model_olustur(model)

    def analiz_yap(self, goruntu_yolu: str):
        """Orijinal goruntu ve Grad-CAM overlay dizisini dondurur."""
        model_girisi, ham_goruntu = goruntu_yukle_ve_hazirla(goruntu_yolu)
        isi_haritasi, _, _ = isi_haritasi_uret(self.grad_model, model_girisi)
        overlay = overlay_olustur(ham_goruntu, isi_haritasi)
        return {
            "orijinal": ham_goruntu.astype(np.uint8),
            "overlay": overlay.astype(np.uint8),
            "katman": self.son_katman_adi,
        }
