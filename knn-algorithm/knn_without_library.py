import pandas as pd


def knn_algorithm(feature_cols, label_col, data, prediction, k):
    """
    Kütüphane kullanmadan KNN algoritması ile tahmin yapar.
    :param feature_cols:
    :param label_col: 
    :param data: 
    :param prediction:
    :param k: 
    :return:
    """
    try:
        distances = calculate_euclid_distance(
            feature_cols, data, prediction
        )
        result_weighted, result_base, distances_k = find_category(
            distances, k, data, label_col
        )
        return result_weighted, result_base,distances_k

    except Exception as e:
        print(e)


def calculate_euclid_distance(feature_cols, data, prediction):
    """
    Tüm verilerle verilen giriş arasındaki Öklid mesafesini hesaplar.
    :param feature_cols:
    :param data:
    :param prediction:
    :return: 
    """
    try:
    
        distances = []

        for i in range(len(data)):
            distance = 0
            for row, col in enumerate(feature_cols):
                distance += (prediction[row] - data[col][i]) ** 2

            distance **= 0.5

            distances.append((i,distance))

        return distances

    except Exception as e:
        print(e)


def find_category(distances, k, data, label_col):
    """
    En yakın k komşuya göre sınıf tahmini yapar. Hem Temel Knn hem de Ağırlıklı Knn kullanılır.
    :param distances:
    :param k:
    :param data:
    :param label_col:
    :return:
    """
    try:
        count_0 = 0
        count_1 = 0

        freq0 = 0
        freq1 = 0
        
        distances_k =[]
        sorted_distances = sorted(distances, key=lambda x: x[1])
        sorted_distances_k = sorted_distances[:k]

        for idx, distance in sorted_distances_k:
            if data[label_col][idx] == 0:
                count_0 += 1
                try:
                    freq0 += 1 / distance
                    distances_k.append((float(distance),int(data[label_col][idx])))
                except ZeroDivisionError:
                    freq0 += float('inf')  
            else:
                count_1 += 1
                try:
                    freq1 += 1 / distance
                    distances_k.append((float(distance),int(data[label_col][idx])))

                except ZeroDivisionError:
                    freq1 += float('inf')

        if freq0 > freq1:
            result_weighted = 0
        else:
            result_weighted = 1
        
        if count_0 > count_1:
            result_base = 0
        else:
            result_base = 1
    except Exception as e:
        print(e)
    return result_weighted, result_base,distances_k



def main():
    
    data = pd.read_csv("iphone_purchase_records.csv")
    feature_cols = data.columns[:-1]
    label_col = data.columns[-1]

    print(f"Özellik sütunları: {list(feature_cols)}")
    print(f"Veri kümesi:\n{data}\n")

    prediction = []
    for col in feature_cols:
        try:
            user_input = float(input(f"Lütfen {col} değerini yazınız: "))
            prediction.append(user_input)
        except Exception as e:
            print(e)
            return

    print(f"Tüm girilen özellik değerleri: {prediction}")

    try:
        k = int(input("k (en yakın komşu) sayısını giriniz:"))
        if k < len(data):
            result_weighted, result_base, distances_k = knn_algorithm(feature_cols, label_col, data, prediction, k)
            print(f"\nTemel Knn ile tahmin edilen kategori: {result_base}\nAğırlıklı Knn ile tahmin edilen kategori :{result_weighted}\n")
            print(f"En yakın 3 uzaklık ve etiketleri{distances_k}")
        else:
            print("k değeri veri boyutundan büyük, lütfen tekrar deneyiniz.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
