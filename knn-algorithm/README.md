# K-En Yakın Komşu (KNN) Algoritması ile iPhone Satın Alma Tahmini

Bu proje, müşterilerin iPhone satın alıp almayacağını tahmin etmek için K-En Yakın Komşu (KNN) algoritmasını kullanmaktadır. Algoritma, hem **scikit-learn** kütüphanesi ile hem de kütüphane kullanılmadan, elle yazılmış haliyle uygulanmıştır.

K değerinin kullanıcı tarafından belirlenmesinin sebebi, modelin karmaşıklığını ve duyarlılığını kontrol edebilmektir. Küçük k değerleri aşırı öğrenmeye (overfitting), büyük k değerleri ise yetersiz öğrenmeye (underfitting) yol açabilir.

---

## KNN Algoritması Hakkında

K-En Yakın Komşu (KNN) algoritması, sınıflandırma ve regresyon problemlerinde kullanılan, basit ve etkili bir makine öğrenimi yöntemidir. Bu projede yalnızca sınıflandırma problemi için uygulanmıştır. Algoritma, yeni bir veri noktasını en yakın **k** komşusunun sınıfına göre sınıflandırır.

---

## Veri Seti

Veri setinde müşterilere ait bilgiler bulunmaktadır:

| Özellik               | Açıklama                             |
|-----------------------|--------------------------------------|
| Age (Yaş)             | Müşterinin yaşı                      |
| Salary (Maaş)         | Müşterinin yıllık geliri             |
| Purchase iPhone       | Hedef değişken (1 = Evet, 0 = Hayır) |

---

## Projede Kullanılan Yöntemler

### 1. Kütüphane Kullanarak KNN (`knn_with_library.py`)

- Python’un **scikit-learn** kütüphanesi kullanılmıştır.  
- `KNeighborsClassifier` sınıfı ile model oluşturulmuş, `.fit()` metodu ile eğitim yapılmıştır.  
- Tahminler `.predict()` metodu ile gerçekleştirilir.

---

### 2. Kütüphane Kullanmadan Yazılmış KNN (`knn_without_library.py`)

- KNN algoritması sıfırdan yazılmıştır.  
- Sadece CSV dosyasını okumak için pandas kütüphanesi kullanılmıştır.  
- Eğitim aşaması olmadan tüm veriler bellekte tutulur.  
- Yeni veri ile tüm veriler arasındaki **Öklidyen mesafe (L2 Normu)** hesaplanır.  
- En yakın **k** komşunun sınıfına göre sınıflandırma yapılır.  
- Hem Temel KNN hem de Ağırlıklı KNN uygulanmıştır.  
- Ağırlıklı KNN, yakın komşulara daha fazla ağırlık vererek daha doğru sonuçlar sağlar.

---

## KNN ve Ağırlıklı KNN Hesaplama Örneği

| LABEL | DISTANCE | WEIGHT (1/DISTANCE) |
|-------|----------|---------------------|
| 1     | 1.0      | 1.000               |
| 0     | 8.0      | 0.125               |
| 0     | 8.0      | 0.125               |

**Temel KNN:**  
- 0 etiketi sayısı = 2  
- 1 etiketi sayısı = 1  
- **Sonuç:** En çok olan etiket (0) seçilir.

**Ağırlıklı KNN:**  
- 0 için toplam ağırlık = 0.125 + 0.125 = 0.25  
- 1 için toplam ağırlık = 1.0  
- **Sonuç:** Daha yüksek toplam ağırlığa sahip etiket (1) seçilir.

---

## Kullanım ve Sonuçlar

Kütüphane kullanılarak yazılan KNN algoritmasının terminal çıktısı:  
![knn_with_library.png](knn_with_library.png)

Kütüphane kullanılmadan yazılan Temel ve Ağırlıklı KNN algoritmalarının terminal çıktısı:  
![knn_without_library.png](knn_without_library.png)

İki farklı KNN algoritmasının tahmin sonuçları ve çalışma süreleri karşılaştırması:  
![test.png](test.png)


