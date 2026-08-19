"""Masaustu uygulama giris noktasi.

Uygulama mimarisi:
- `ui/main_window.py`: Arayuz ve kullanici akis yonetimi
- `utils/predictor.py`: Tahmin servis katmani
- `utils/gradcam.py`: Grad-CAM servis katmani
- `utils/image_utils.py`: Goruntu/gosterim yardimcilari
"""

import sys

# KRITIK: Windows'ta Qt baslatildiktan sonra TensorFlow importu bazen
# DLL yukleme hatasi verebiliyor. Bu nedenle TensorFlow'u once yukluyoruz.
import tensorflow as tf  # noqa: F401
from PyQt5.QtWidgets import QApplication

from ui.main_window import AnaPencere


def main():
    """PyQt5 uygulamasini baslatir."""
    uygulama = QApplication(sys.argv)
    pencere = AnaPencere()
    pencere.show()
    sys.exit(uygulama.exec_())


if __name__ == "__main__":
    main()
