import pandas as pd


def knn_algorithm(feature_cols, label_col, data, prediction, k):
    """
    En yakın komşulara göre sınıf tahmini yapar.
    :param feature_cols:
    :param label_col: 
    :param data: 
    :param prediction:
    :param k: 
    :return:
    """
    try:
        category_0, category_1 = calculate_euclid_distance(
            feature_cols, label_col, data, prediction
        )
        category_result = find_category(
            category_0, category_1, k, data, label_col
        )
        return category_result

    except Exception as e:
        print(e)


def calculate_euclid_distance(feature_cols, label_col, data, prediction):
    """
    Tüm verilerle verilen giriş arasındaki Öklid mesafesini hesaplar.
    :param feature_cols:
    :param label_col:
    :param data:
    :param prediction:
    :return: 
    """
    try:
        category_1 = []
        category_0 = []

        for i in range(len(data)):
            distance = 0
            for row, col in enumerate(feature_cols):
                distance += (prediction[row] - data[col][i]) ** 2

            distance **= 0.5

            if data[label_col][i] == 1:
                category_1.append((i, distance))
            else:
                category_0.append((i, distance))

        return category_0, category_1

    except Exception as e:
        print(e)


def find_category(category_0, category_1, k, data, label_col):
    """
    En yakın k komşuya göre sınıf tahmini yapar.
    :param category_0:
    :param category_1:
    :param k:
    :param data:
    :param label_col:
    :return:
    """
    try:
        count_0 = 0
        count_1 = 0
        total_category = category_0 + category_1
        total_sorted = sorted(total_category, key=lambda x: x[1])
        total_k = total_sorted[:k]

        for idx, _ in total_k:
            if data[label_col][idx] == 0:
                count_0 += 1
            else:
                count_1 += 1

        if count_0 > count_1:
            return 0
        else:
            return 1

    except Exception as e:
        print(e)


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
        k = int(input("k (en yakın komşu) sayısını giriniz: "))
        if k < len(data):
            result = knn_algorithm(feature_cols, label_col, data, prediction, k)
            print(f"Tahmin edilen kategori: {result}")
        else:
            print("k değeri veri boyutundan büyük, lütfen tekrar deneyiniz.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
