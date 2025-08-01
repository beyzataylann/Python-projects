K-En Yakın Komşu (KNN) Algoritması ile iPhone Satın Alma Tahmini
Bu proje, müşterilerin iPhone satın alıp almayacağını tahmin etmek için K-En Yakın Komşu (KNN) algoritmasını kullanmaktadır. KNN algoritması hem kütüphane kullanılarak hem de sıfırdan elle yazılarak uygulanmıştır.

KNN Algoritması Hakkında
K-En Yakın Komşu (KNN) algoritması, hem sınıflandırma hem de regresyon problemlerinde kullanılan, basit ve etkili bir makine öğrenimi yöntemidir.
Sınıflandırma: Yeni bir veri noktası, en yakın k komşusunun sınıfına göre sınıflandırılır.
Regresyon: Yeni veri noktasının değeri, en yakın k komşusunun değerlerinin ortalaması alınarak tahmin edilir.

Bu projede, yalnızca sınıflandırma problemi için KNN algoritması kullanılmıştır.

Veri Seti Hakkında
Veri setinde müşteri bilgileri yer almaktadır ve şu özelliklerden oluşur:
Age (Yaş): Müşterinin yaşı
Salary (Maaş): Müşterinin yıllık maaşı
Purchase Iphone (iPhone Satın Alma): Hedef değişken; müşterinin iPhone satın alıp almadığını belirtir (1 = Evet, 0 = Hayır)

Amacımız, bu özellikler üzerinden yeni müşterilerin iPhone satın alma durumunu tahmin etmektir.
Projede Kullanılan Yöntemler
Projede iki farklı KNN uygulaması yer almaktadır:

1. Kütüphane Kullanarak KNN
Python’un güçlü makine öğrenimi kütüphanesi scikit-learn kullanılmıştır. KNeighborsClassifier sınıfı ile kolayca model oluşturulup eğitilmiştir. Bu yöntem, kodu kısa, temiz ve performansı yüksek hale getirir. 
Model .fit() metodu ile eğitim verisini belleğe alır ve .predict() ile tahmin yapar.

2. Kütüphane Kullanmadan Elle Yazılmış KNN
KNN algoritmasının mantığını daha iyi anlamak için sıfırdan elle yazılmıştır. Eğitim süreci olmadan, tüm veri bellekte saklanır. Yeni bir veri noktası ile tüm veriler arasındaki Öklidyen mesafesi hesaplanır. En yakın k komşunun sınıfına göre çoğunluk oyu ile sınıflandırma yapılır.

Kullanım
Kütüphane kullanılmadan yazılan KNN algoritmasının terminal çıktısı aşağıdadır:
<img width="932" height="109" alt="image" src="https://github.com/user-attachments/assets/b6e60df2-500f-4525-b298-966a315d353d" />

Kütüphane kullanılarak yazılan KNN algoritmasının terminal çıktısı aşağıdadır:
<img width="948" height="126" alt="image" src="https://github.com/user-attachments/assets/33388257-d567-4cb7-a814-216d1b231ec9" />


