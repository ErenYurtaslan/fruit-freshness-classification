# Mendeley Sonrasi Dogrulama Raporu

*Olusturma: 2026-06-11 15:14*

## 7 Dis Ornek (outputs/ood_test)

| Ornek | Durum | Mesafe | Tahmin |
|---|---|---|---|
| ejder_meyvesi | Sinirda (Fresh Apple (?)) | 0.482 | freshapples |
| greyfurt | Sinirda (Fresh Orange (?)) | 0.463 | freshoranges |
| lici | Sinirda (Fresh Apple (?)) | 0.407 | freshapples |
| mandalina | Bilinen (Fresh Orange) | 0.327 | freshoranges |
| mor_erik | Sinirda (Fresh Apple (?)) | 0.446 | freshapples |
| mor_uzum | Sinirda (Rotten Strawberry (?)) | 0.370 | rottenstrawberry |
| murdum_erigi | Bilinmeyen | 0.485 | rottenbanana |

**Isaretleme:** 6/7 (hedef >= 5/7)

## ood_eval Holdout (taze/curuk ayri sinif klasorleri)

| Sinif | N test | Bilinmeyen | Sinirda | Bilinen |
|---|---|---|---|---|
| freshgrapes | 10 | 0 | 0 | 10 |
| freshguava | 10 | 0 | 0 | 10 |
| freshjujube | 10 | 0 | 0 | 10 |
| freshpomegranate | 10 | 0 | 0 | 10 |
| freshstrawberry | 10 | 0 | 0 | 10 |
| rottengapes | 10 | 0 | 0 | 10 |
| rottenguava | 10 | 0 | 0 | 10 |
| rottenjujube | 10 | 0 | 0 | 10 |
| rottenpomegranate | 10 | 0 | 0 | 10 |
| rottenstrawberry | 10 | 0 | 0 | 10 |

**ood_eval holdout tanima:** %100.0 bilinen (100/100), yanlis bilinmeyen %0.0

## Gercek Dunya Hedef Ornekleri

- **sari_elma:** dosya bulunamadi (atlandi)
- **curuk_portakal:** dosya bulunamadi (atlandi)

## Ozet

- Uc kademeli OOD karari ve sinirda UI yumusatmasi (Belirsiz) aktif.
- Mendeley gercek dunya gorselleri egitime entegre edildi; model test dogrulugu ~%96.