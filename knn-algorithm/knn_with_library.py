import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def main():

        data = pd.read_csv("iphone_purchase_records.csv")
        X = data[['Age', 'Salary']]
        y = data['Purchase Iphone']
        
        p_age = float(input("Lütfen yaşınızı yazınız: "))
        p_salary = float(input("Lütfen maaşınızı yazınız: "))
        k = int(input("k (en yakın komşu) sayısını giriniz: "))

        
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X, y)

        input_d = pd.DataFrame([[p_age, p_salary]], columns=['Age', 'Salary'])

        prediction = model.predict(input_d)
        print(f"Kategori: {(prediction[0])}")


if __name__ == "__main__":
    main()
