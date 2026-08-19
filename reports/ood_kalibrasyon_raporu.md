# OOD Eşik Kalibrasyon Raporu (Üç Kademeli Karar)

*Oluşturma zamanı: 2026-06-11 15:14*

## Karar Modeli

- `mesafe > T_yüksek` → **Bilinmeyen** (koşulsuz)
- `mesafe < T_düşük` → **Bilinen**
- Gri bölge → softmax sınıfı ile en yakın merkez sınıfı uyuşuyor ve
  güven yüksekse **Sınırda** (tahmin düşük güven uyarısıyla gösterilir),
  aksi halde Bilinmeyen

## Kısıtlar

1. Bilinen test örneklerinde yanlış 'bilinmeyen' oranı **≤ %2**
2. 7 dış örnekte işaretleme (kesin bilinmeyen + sınırda uyarısı) **≥ 5/7**
3. ood_eval holdout setinde (egitimdeki yeni siniflar, taze/curuk ayri)
   yanlis bilinmeyen orani **≤ %10**

Bilinen örneklem: **1344** test görseli (84/sınıf). Holdout örneklemi: **200** görsel.

## Mesafe Dağılımı Yüzdelikleri

| Küme | N | P50 | P90 | P95 | P99 | Maks |
|---|---|---|---|---|---|---|
| Bilinen test | 1344 | 0.232 | 0.313 | 0.338 | 0.427 | 0.511 |
| 7 dış örnek | 7 | 0.446 | 0.483 | 0.484 | 0.485 | 0.485 |
| ood_eval/Fresh Grape | 20 | 0.212 | 0.338 | 0.348 | 0.465 | 0.494 |
| ood_eval/Fresh Guava | 20 | 0.144 | 0.209 | 0.214 | 0.215 | 0.215 |
| ood_eval/Fresh Jujube | 20 | 0.162 | 0.323 | 0.325 | 0.344 | 0.349 |
| ood_eval/Fresh Pomegranate | 20 | 0.224 | 0.281 | 0.292 | 0.318 | 0.325 |
| ood_eval/Fresh Strawberry | 20 | 0.209 | 0.321 | 0.334 | 0.448 | 0.477 |
| ood_eval/Rotten Gape | 20 | 0.226 | 0.293 | 0.316 | 0.343 | 0.349 |
| ood_eval/Rotten Guava | 20 | 0.237 | 0.298 | 0.319 | 0.336 | 0.340 |
| ood_eval/Rotten Jujube | 20 | 0.264 | 0.331 | 0.348 | 0.348 | 0.349 |
| ood_eval/Rotten Pomegranate | 20 | 0.247 | 0.305 | 0.316 | 0.332 | 0.336 |
| ood_eval/Rotten Strawberry | 20 | 0.242 | 0.308 | 0.339 | 0.341 | 0.342 |

## Eşik Izgarası (uygun adaylar + mevcut config)

| T_düşük | T_yüksek | Bilinen→yanlış bilinmeyen | Bilinen→sınırda | Dış: kesin | Dış: sınırda | ood_eval yanlış bilinmeyen | Uygun |
|---|---|---|---|---|---|---|---|
| 0.355 | 0.440 | %1.93 | %1.19 | 4/7 | 2/7 | %1.0 | EVET |
| 0.355 | 0.450 | %1.79 | %1.34 | 3/7 | 3/7 | %1.0 | EVET |
| 0.355 | 0.460 | %1.79 | %1.34 | 3/7 | 3/7 | %1.0 | EVET |
| 0.355 | 0.470 | %1.71 | %1.41 | 2/7 | 4/7 | %1.0 | EVET |
| 0.355 | 0.480 | %1.71 | %1.41 | 2/7 | 4/7 | %0.5 | EVET |
| 0.355 | 0.490 | %1.71 | %1.41 | 1/7 | 5/7 | %0.5 | EVET |
| 0.355 | 0.500 | %1.71 | %1.41 | 1/7 | 5/7 | %0.5 | EVET |
| 0.355 | 0.510 | %1.71 | %1.41 | 1/7 | 5/7 | %0.5 | EVET |
| 0.355 | 0.520 | %1.71 | %1.41 | 1/7 | 5/7 | %0.5 | EVET |
| 0.355 | 0.530 | %1.71 | %1.41 | 1/7 | 5/7 | %0.5 | EVET |
| 0.355 | 0.540 | %1.71 | %1.41 | 1/7 | 5/7 | %0.5 | EVET |
| 0.355 | 0.550 | %1.71 | %1.41 | 1/7 | 5/7 | %0.5 | EVET |
| 0.355 | 0.560 | %1.71 | %1.41 | 1/7 | 5/7 | %0.5 | EVET |
| 0.380 | 0.390 | %1.86 | %0.30 | 5/7 | 0/7 | %1.0 | EVET |
| 0.380 | 0.400 | %1.79 | %0.37 | 5/7 | 0/7 | %1.0 | EVET |
| 0.380 | 0.410 | %1.79 | %0.37 | 4/7 | 1/7 | %1.0 | EVET |
| 0.380 | 0.420 | %1.71 | %0.45 | 4/7 | 1/7 | %1.0 | EVET |
| 0.380 | 0.430 | %1.64 | %0.52 | 4/7 | 1/7 | %1.0 | EVET |
| 0.380 | 0.440 | %1.56 | %0.60 | 4/7 | 1/7 | %1.0 | EVET |
| 0.380 | 0.450 | %1.41 | %0.74 | 3/7 | 2/7 | %1.0 | EVET |
| 0.380 | 0.460 | %1.41 | %0.74 | 3/7 | 2/7 | %1.0 | EVET |
| 0.380 | 0.470 | %1.34 | %0.82 | 2/7 | 3/7 | %1.0 | EVET |
| 0.380 | 0.480 | %1.34 | %0.82 | 2/7 | 3/7 | %0.5 | EVET |
| 0.380 | 0.490 | %1.34 | %0.82 | 1/7 | 4/7 | %0.5 | EVET |
| 0.380 | 0.500 | %1.34 | %0.82 | 1/7 | 4/7 | %0.5 | EVET |
| 0.380 | 0.510 | %1.34 | %0.82 | 1/7 | 4/7 | %0.5 | EVET |
| 0.380 | 0.520 | %1.34 | %0.82 | 1/7 | 4/7 | %0.5 | EVET |
| 0.380 | 0.530 | %1.34 | %0.82 | 1/7 | 4/7 | %0.5 | EVET |
| 0.380 | 0.540 | %1.34 | %0.82 | 1/7 | 4/7 | %0.5 | EVET |
| 0.380 | 0.550 | %1.34 | %0.82 | 1/7 | 4/7 | %0.5 | EVET |
| 0.380 | 0.560 | %1.34 | %0.82 | 1/7 | 4/7 | %0.5 | EVET |
| 0.405 | 0.410 | %1.49 | %0.00 | 4/7 | 1/7 | %1.0 | EVET |
| 0.405 | 0.420 | %1.41 | %0.07 | 4/7 | 1/7 | %1.0 | EVET |
| 0.405 | 0.430 | %1.34 | %0.15 | 4/7 | 1/7 | %1.0 | EVET |
| 0.405 | 0.440 | %1.26 | %0.22 | 4/7 | 1/7 | %1.0 | EVET |
| 0.405 | 0.450 | %1.12 | %0.37 | 3/7 | 2/7 | %1.0 | EVET |
| 0.405 | 0.460 | %1.12 | %0.37 | 3/7 | 2/7 | %1.0 | EVET |
| 0.405 | 0.470 | %1.04 | %0.45 | 2/7 | 3/7 | %1.0 | EVET |
| 0.405 | 0.480 | %1.04 | %0.45 | 2/7 | 3/7 | %0.5 | EVET |
| 0.405 | 0.490 | %1.04 | %0.45 | 1/7 | 4/7 | %0.5 | EVET |
| 0.405 | 0.500 | %1.04 | %0.45 | 1/7 | 4/7 | %0.5 | EVET |
| 0.405 | 0.510 | %1.04 | %0.45 | 1/7 | 4/7 | %0.5 | EVET |
| 0.405 | 0.520 | %1.04 | %0.45 | 1/7 | 4/7 | %0.5 | EVET |
| 0.405 | 0.530 | %1.04 | %0.45 | 1/7 | 4/7 | %0.5 | EVET |
| 0.405 | 0.540 | %1.04 | %0.45 | 1/7 | 4/7 | %0.5 | EVET |
| 0.405 | 0.550 | %1.04 | %0.45 | 1/7 | 4/7 | %0.5 | EVET |
| 0.405 | 0.560 | %1.04 | %0.45 | 1/7 | 4/7 | %0.5 | EVET |

## Seçilen Eşikler (config)

- `OOD_MESAFE_T_YUKSEK = 0.485`
- `OOD_MESAFE_T_DUSUK = 0.365`
- `OOD_GRI_MAXPROB_ESIK = 0.8`
- `OOD_GRI_ENTROPY_ESIK = 0.3`


## Holdout Seti Ayrıntısı (mevcut eşiklerle)

| Sınıf | N | Bilinmeyen | Sınırda | Bilinen |
|---|---|---|---|---|
| Fresh Grape | 20 | 1 | 0 | 19 |
| Fresh Guava | 20 | 0 | 0 | 20 |
| Fresh Jujube | 20 | 0 | 0 | 20 |
| Fresh Pomegranate | 20 | 0 | 0 | 20 |
| Fresh Strawberry | 20 | 0 | 1 | 19 |
| Rotten Gape | 20 | 0 | 0 | 20 |
| Rotten Guava | 20 | 0 | 0 | 20 |
| Rotten Jujube | 20 | 0 | 0 | 20 |
| Rotten Pomegranate | 20 | 0 | 0 | 20 |
| Rotten Strawberry | 20 | 0 | 0 | 20 |

## Doğruluk Etkisi

- Örneklemde ham model doğruluğu (OOD yok): **%85.34**
- Gösterilen (bilinen + sınırda) örneklerde doğruluk: **%85.66**
- Bilinen örneklerde sınırda uyarısı oranı: **%0.97**

## Not

Eğitim setine Mendeley gerçek dünya görselleri (taze/çürük elma, muz,
portakal) eklendiği için bilinen bölgenin mesafe dağılımı önceki modele
göre değişmiştir; eşikler bu yeni dağılıma göre taranmıştır. Üç kademeli
karar yapısı korunmuştur.