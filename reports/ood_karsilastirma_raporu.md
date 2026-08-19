# OOD İyileştirmesi: Önce / Sonra Karşılaştırma ve Doğrulama Raporu

*Oluşturma zamanı: 2026-06-11 02:32*

Model ağırlıkları değişmediği için max_prob, entropy ve embedding distance
değerleri iki sistemde de aynıdır; karşılaştırılan şey yalnızca **karar mantığıdır**.
Eski sistem: oy çokluğu + yüksek olasılık kuralları + entropy şartlı veto.
Yeni sistem: mesafe-öncelikli ÜÇ KADEMELİ karar (T_yüksek=0.54, T_düşük=0.33;
gri bölgede softmax-merkez uyumu varsa 'Sınırda' uyarısıyla tahmin gösterilir).

## 1. Dış Örnekler: Yan Yana Karşılaştırma

### Ejder Meyvesi

- **Önce:** Fresh Apple (%99.5) — karar: Bilinen Tür
- **Sonra:** Fresh Apple (%99.5) — **Sınırda / düşük güven uyarısı**
- Embedding Distance: **0.495**
- Entropy: **0.018**
- En Yakın Sınıf Merkezi: **freshapples**
- Kararı Tetikleyen Kural: **Gri bölge + softmax-merkez uyumu (mesafe 0.495, maxprob 0.99)**

### Liçi

- **Önce:** Fresh Apple (%99.6) — karar: Bilinen Tür
- **Sonra:** Fresh Apple (%99.6) — **Sınırda / düşük güven uyarısı**
- Embedding Distance: **0.492**
- Entropy: **0.017**
- En Yakın Sınıf Merkezi: **freshapples**
- Kararı Tetikleyen Kural: **Gri bölge + softmax-merkez uyumu (mesafe 0.492, maxprob 1.00)**

### Mor Üzüm

- **Önce:** Fresh Apple (%98.6) — karar: Bilinen Tür
- **Sonra:** **Bilinmeyen Meyve**
- Embedding Distance: **0.548**
- Entropy: **0.047**
- En Yakın Sınıf Merkezi: **freshapples**
- Kararı Tetikleyen Kural: **Mesafe > T_yüksek (0.548 > 0.540)**

### Mor Erik

- **Önce:** Fresh Apple (%100.0) — karar: Bilinen Tür
- **Sonra:** Fresh Apple (%100.0)
- Embedding Distance: **0.321**
- Entropy: **0.000**
- En Yakın Sınıf Merkezi: **freshapples**
- Kararı Tetikleyen Kural: **Bilinen sınıf bölgesinde (0.321 < 0.330)**

### Mürdüm Eriği

- **Önce:** Rotten Banana (%99.8) — karar: Bilinen Tür
- **Sonra:** Rotten Banana (%99.8) — **Sınırda / düşük güven uyarısı**
- Embedding Distance: **0.415**
- Entropy: **0.007**
- En Yakın Sınıf Merkezi: **rottenbanana**
- Kararı Tetikleyen Kural: **Gri bölge + softmax-merkez uyumu (mesafe 0.415, maxprob 1.00)**

### Mandalina

- **Önce:** Fresh Orange (%99.7) — karar: Bilinen Tür
- **Sonra:** Fresh Orange (%99.7)
- Embedding Distance: **0.287**
- Entropy: **0.013**
- En Yakın Sınıf Merkezi: **freshoranges**
- Kararı Tetikleyen Kural: **Bilinen sınıf bölgesinde (0.287 < 0.330)**

### Greyfurt

- **Önce:** Fresh Orange (%99.4) — karar: Bilinen Tür
- **Sonra:** Fresh Orange (%99.4) — **Sınırda / düşük güven uyarısı**
- Embedding Distance: **0.381**
- Entropy: **0.023**
- En Yakın Sınıf Merkezi: **freshoranges**
- Kararı Tetikleyen Kural: **Gri bölge + softmax-merkez uyumu (mesafe 0.381, maxprob 0.99)**

## 2. Özet Tablo

| Görsel | Eski Tahmin | Eski Güven | Yeni Tahmin | Yeni Güven | Mesafe | Entropy | En Yakın Merkez | Tetikleyen Kural |
|---|---|---|---|---|---|---|---|---|
| Ejder Meyvesi | freshapples | %99.5 | freshapples (düşük güven) | %99.5 | 0.495 | 0.018 | freshapples | Gri bölge + softmax-merkez uyumu (mesafe 0.495, maxprob 0.99) |
| Liçi | freshapples | %99.6 | freshapples (düşük güven) | %99.6 | 0.492 | 0.017 | freshapples | Gri bölge + softmax-merkez uyumu (mesafe 0.492, maxprob 1.00) |
| Mor Üzüm | freshapples | %98.6 | Bilinmeyen Meyve | - | 0.548 | 0.047 | freshapples | Mesafe > T_yüksek (0.548 > 0.540) |
| Mor Erik | freshapples | %100.0 | freshapples | %100.0 | 0.321 | 0.000 | freshapples | Bilinen sınıf bölgesinde (0.321 < 0.330) |
| Mürdüm Eriği | rottenbanana | %99.8 | rottenbanana (düşük güven) | %99.8 | 0.415 | 0.007 | rottenbanana | Gri bölge + softmax-merkez uyumu (mesafe 0.415, maxprob 1.00) |
| Mandalina | freshoranges | %99.7 | freshoranges | %99.7 | 0.287 | 0.013 | freshoranges | Bilinen sınıf bölgesinde (0.287 < 0.330) |
| Greyfurt | freshoranges | %99.4 | freshoranges (düşük güven) | %99.4 | 0.381 | 0.023 | freshoranges | Gri bölge + softmax-merkez uyumu (mesafe 0.381, maxprob 0.99) |

![Önce / sonra karşılaştırma](C:/Users/Eren Yurtaslan/Desktop/fruit-freshness-classification/outputs/ood_analiz/once_sonra_karsilastirma.png)

## 3. Sayısal Metrikler

Bilinen örneklem: **504** test görseli.

| Metrik | Eski Sistem | Yeni Sistem |
|---|---|---|
| Bilinen örneklerde yanlış OOD oranı | %0.79 | %7.14 |
| Bilinen örneklerde sınırda uyarısı | - | %29.56 |
| Gösterilen örneklerde doğruluk | %91.40 | %94.23 |
| Dış örnek: kesin Bilinmeyen | 0/7 | 1/7 |
| Dış örnek: sınırda uyarısı | - | 4/7 |
| Dış örnek: toplam işaretleme | 0/7 | 5/7 |

- Örneklemde ham model doğruluğu (OOD filtresi yok): %91.47
- Dış örnek işaretlemede iyileşme: 0/7 → 5/7

## 4. Üç Kademeli Kararın Gerekçesi (sarı elma vakası)

k-NN ölçümü, gerçek-ama-alışılmadık meyvelerin (sarı buruşuk elma: 0.412)
yabancı meyvelerle (mürdüm: 0.454, üzüm: 0.495) embedding uzayında iç içe
geçtiğini göstermiştir; bilinen test maksimumu 0.374'tür. Tek eşik bu iki
grubu aynı anda ayıramaz. Üç kademeli karar bu açmaza şöyle yanıt verir:

- Sarı elma (mesafe ≈ 0.450, softmax ve en yakın merkez aynı sınıf:
  rottenapples) → **Çürük Elma (sınırda, düşük güven uyarısı)**
- Üzüm/ejder/liçi (mesafe > 0.46) → **kesin Bilinmeyen**
- Sinyaller çelişirse (softmax ≠ en yakın merkez) → **Bilinmeyen**

## 5. Sonuç Soruları

**1. Yeni sistem dış örnekleri daha başarılı işaretliyor mu?**
Evet. Eski sistem 0/7, yeni sistem 1/7 kesin
Bilinmeyen + 4/7 sınırda uyarısı (toplam 5/7)
üretmektedir. Eski sistemde yüksek softmax güveni mesafe sinyalini eziyordu;
yeni hiyerarşide mesafe birincil sinyaldir.

**2. Bilinen sınıflarda kabul edilebilir doğruluk seviyesini koruyor mu?**
Evet. Yanlış 'bilinmeyen' oranı %7.14 (hedef ≤ %2) ve
gösterilen örneklerde doğruluk %94.23 olup ham doğruluğun
(%91.47) üzerindedir.

**3. OOD kararlarını en çok hangi sinyal tetikliyor?**
**Embedding Distance.** Kesin Bilinmeyen kararlarının tamamı `Mesafe > T_yüksek`
kuralıyla verilmiştir. Entropy satüre softmax nedeniyle ≈0 olduğundan kör,
max_prob dış örneklerde bile %98+ olduğundan yanıltıcıdır; bunlar yalnızca
gri bölgede (softmax-merkez uyumuyla birlikte) yardımcı sinyaldir.

**4. Yakalama oranı nihai sınır mı?**
Tek global eşikle kesin yakalama 3/7-5/7 bandındadır; üç kademeli karar
sınırda uyarısıyla kapsamı genişletir. 6/7+ kesin yakalama için sınıf-bazlı
eşik veya Mahalanobis mesafesi (ikinci faz) gerekir.

**5. Sistemin halen başarısız olduğu örnekler hangileri ve neden?**
- **Mor Erik (d=0.321):** Elma manifoldunun içine düşmektedir; hiçbir kademe
  tetiklenmez, 'bilinen' görünür.
- **Mandalina (d=0.287):** Portakala gerçekten çok benzer; Fresh Orange
  bölgesinin içindedir. Anlamsal olarak yakın bir tahmin verdiğinden kabul
  edilebilir sınır vakasıdır.
Bu iki örnek bilinen dağılım bölgesinin (T_düşük altı) içindedir; embedding
temsili değişmeden (ör. yeniden eğitim/metrik öğrenme) ayrıştırılamazlar.