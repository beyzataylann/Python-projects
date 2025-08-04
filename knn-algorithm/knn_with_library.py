import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def sklearn_model_predict(age, salary, k, csv_path="iphone_purchase_records.csv"):
    data = pd.read_csv(csv_path)
    X = data[['Age', 'Salary']]
    y = data['Purchase Iphone']

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)

    input_d = pd.DataFrame([[age, salary]], columns=['Age', 'Salary'])
    prediction = model.predict(input_d)
    return prediction[0]

def main():
    try:
        p_age = float(input("Lütfen yaşınızı yazınız: "))
        p_salary = float(input("Lütfen maaşınızı yazınız: "))
        k = int(input("k (en yakın komşu) sayısını giriniz: "))
        result = sklearn_model_predict(p_age, p_salary, k)
        print(f"Kategori: {result}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
