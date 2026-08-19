# Durum Analizi: OOD Katmanı vs Veri Seti vs Mendeley Entegrasyonu

*Oluşturma: 2026-06-11*

## Nereden geldik?

1. **Kapalı küme sınıflandırıcı** (6 sınıf: taze/çürük elma, muz, portakal) eğitildi.
2. Dış meyveler (ejder, üzüm, mürdüm) **%98+ güvenle yanlış sınıfa** atanıyordu → **OOD katmanı** eklendi.
3. OOD üç kademeli karara evrildi: Bilinen / Sınırda / Bilinmeyen (embedding mesafesi + softmax).
4. **Mendeley planı** uygulandı; ancak zip yalnızca `Augmented Image` içeriyordu, `Original Image` yoktu.
5. `integrate_mendeley.py` ilk çalışmada ~200 görsel/sınıf (`mendeley_*`) ekledi; tam artırılmış set (~740/sınıf) **egitime girmedi**.
6. OOD eşikleri dış örnekleri yakalamak için genişletildi → **veri setinden gelen görseller** gri bölgeye düşüp **%100 güvenle bile "Sınırda / düşük güvenilirlik"** uyarısı alıyor.

## Ekran görüntülerindeki hatalar

| Görüntü | Model | OOD kararı | Sorunun kökü |
|---|---|---|---|
| Veri seti çürük muz (%100) | Rotten Banana | Sınırda + uyarı | Mesafe gri bölgede; OOD eğitim dağılımını temsil etmiyor |
| Ejder meyvesi (%99) | Fresh Apple | Sınırda + Elma/TAZE | Yüksek softmax override; yabancı meyve eğitimde yok |
| Veri seti çürük elma (%65 taze) | Fresh Apple | Bilinen (yanlış sınıf) | Az/çeşitli çürük elma örneği; model hatası |
| Veri seti mandalina benzeri | Fresh Orange | Sınırda (önceki sürüm) | Aynı OOD gri bölge sorunu |

**Ortak nokta:** Arayüzdeki çelişki "manuel metin" değil; **model softmax çıktısı** ile **OOD embedding kararının** farklı hedefler için tasarlanmış olması. OOD dış dünya için koruma; veri seti örnekleri için gereksiz uyarı üretiyor.

## Eksik parça: temp_dataset kullanılmadı

`temp_dataset/Augmented Image` içinde **~4.500 elma/muz/portakal görseli** var; dataset'te yalnızca **~1.200 mendeley_*** kopyası mevcut. Yabancı meyveler (üzüm, guava, …) bilinçli olarak **egitime girmedi** — yalnızca `outputs/ood_eval/` altında.

## Yapılacaklar (bu oturum)

1. **Tam birleştirme:** Augmented Image → `dataset/train|test` (6 sınıf, %80/20, `mendeley_aug_*`).
2. **Veri seti OOD bypass:** `dataset/` altından yüklenen görsellerde OOD uyarısı kapatılır; yalnızca model tahmini gösterilir.
3. **Yeniden eğitim** genişletilmiş veri ile.
4. **OOD yeniden kalibrasyon** eğitim sonrası.
5. `temp_dataset/` silinir (birleştirme sonrası).

## Beklenti

- Veri setinden seçilen görseller: **eski gibi net** tahmin (uyarı yok).
- Dış dünya elma/muz/portakal: Mendeley çeşitliliğiyle **daha doğru tazelik**.
- Ejder/mürdüm vb.: OOD ile **Bilinmeyen** veya tutarlı sınırda uyarısı.
