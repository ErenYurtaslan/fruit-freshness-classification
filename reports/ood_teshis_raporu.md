# OOD Teşhis Raporu

*Oluşturma zamanı: 2026-06-10 14:58*

## 1. Amaç

Bu rapor, modelin eğitim dağılımı dışındaki (OOD) meyveleri neden
yüksek güvenle bilinen sınıflara atadığını kanıta dayalı olarak belgeler.
Hiçbir model parametresi değiştirilmemiştir; yalnızca mevcut davranış ölçülmüştür.

## 2. Bilinen Dağılım Mesafe İstatistikleri

- Ölçülen bilinen test örneği sayısı: **360**
- Medyan mesafe (P50): **0.2343**
- P90: **0.3230**
- P95: **0.3503**
- P99: **0.3836**
- Mevcut mesafe eşiği (train P95): **0.3548**

![Mesafe dağılımı](C:/Users/Eren Yurtaslan/Desktop/fruit-freshness-classification/outputs/ood_analiz/mesafe_dagilimi.png)

## 3. Dış Örnek Analizi (Mevcut Sistem)

| Örnek | Tahmin | MaxProb | Entropy | Mesafe | En Yakın Merkez | Eski Karar |
|---|---|---|---|---|---|---|
| Ejder Meyvesi | freshapples | %99.5 | 0.018 | 0.495 | freshapples (0.495) | Bilinen Tür |
| Liçi | freshapples | %99.6 | 0.017 | 0.492 | freshapples (0.492) | Bilinen Tür |
| Mor Üzüm | freshapples | %98.6 | 0.047 | 0.548 | freshapples (0.548) | Bilinen Tür |
| Mor Erik | freshapples | %100.0 | 0.000 | 0.321 | freshapples (0.321) | Bilinen Tür |
| Mürdüm Eriği | rottenbanana | %99.8 | 0.007 | 0.415 | rottenbanana (0.415) | Bilinen Tür |
| Mandalina | freshoranges | %99.7 | 0.013 | 0.287 | freshoranges (0.287) | Bilinen Tür |
| Greyfurt | freshoranges | %99.4 | 0.023 | 0.381 | freshoranges (0.381) | Bilinen Tür |

### Sınıf çekim analizi

Dış örneklerin en yakın olduğu sınıf merkezleri:

- **freshapples**: 4 dış örnek
- **freshoranges**: 2 dış örnek
- **rottenbanana**: 1 dış örnek

Apple sınıfları veri setinin yaklaşık %37'sini oluşturduğundan ve
yuvarlak/parlak nesneler apple manifolduna çekildiğinden, dış örneklerin
çoğunun elma merkezlerine yaklaşması veri seti yanlılığıyla (dataset bias) tutarlıdır.

## 4. Embedding Uzayı Görselleştirmesi

![Embedding PCA](C:/Users/Eren Yurtaslan/Desktop/fruit-freshness-classification/outputs/ood_analiz/embedding_pca.png)

PCA izdüşümünde apple kümeleri geniş bir bölgeye yayılmaktadır;
dış örnekler bu bölgenin kenarına düşmekte ancak softmax bunlara
yüksek olasılık atamaktadır. Bu, kapalı küme (closed-set) sınıflandırmanın
yapısal sınırlamasıdır: softmax her girdiyi mevcut 6 sınıftan birine dağıtmak zorundadır.

## 5. Grad-CAM Analizi

![Grad-CAM ızgarası](C:/Users/Eren Yurtaslan/Desktop/fruit-freshness-classification/outputs/ood_analiz/gradcam_izgara.png)

Isı haritaları, modelin dış örneklerde de nesnenin yuvarlak/parlak gövdesine
odaklandığını gösterir. Model 'bu bir elma mı?' sorusunu değil,
'6 sınıftan hangisine en çok benziyor?' sorusunu cevaplamaktadır.

## 6. Temel Bulgular

1. **Mesafe sinyali çalışıyor:** 5/7 dış örnek mevcut
   mesafe eşiğinin (0.355) üzerindedir; embedding uzaklığı dış örnekleri ayırt edebilmektedir.
2. **Karar hiyerarşisi mesafeyi eziyor:** Eski sistem yalnızca 0/7 dış örneği yakalamıştır. `yuksek_olasilik_bilinen` ve `oy_coklugu_bilinen` kuralları, mesafe sinyali
   'bilinmeyen' derken yüksek softmax güveni nedeniyle kararı 'bilinen'e çevirmektedir.
3. **Veto kuralı kör:** Veto `entropy > 0.33` şartına bağlıdır; kendinden emin
   yanlışlarda entropy ≈ 0 olduğu için veto hiç tetiklenmemektedir.
4. **Sorun calibration değil:** Temperature scaling argmax'ı değiştirmez;
   ejder meyvesi yine 'Fresh Apple' kalır, yalnızca güven yüzdesi düşer.

## 7. Önerilen Çözüm

Mesafe-öncelikli hiyerarşik karar:

- `mesafe > T_yüksek` → **Bilinmeyen** (softmax ne derse desin)
- `mesafe < T_düşük` → **Bilinen**
- Gri bölge → max_prob / entropy yardımcı sinyal

Eşikler bilinen test dağılımından kalibre edilir: bilinen örneklerde yanlış
'bilinmeyen' oranı ≤ %2, dış örneklerde yakalama ≥ 5/7 hedeflenir.