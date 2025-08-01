import pandas as pd 


def knn_algorithm(p_age, p_salary, k, data ):
   """
   :param: p_age
   :param: p_salary
   :param: k
   :return:
   """
   try:
        result, category_0, category_1 = calculate_oklid_distance(p_age, p_salary, k, data)
        category_result = find_category(result, category_0, category_1)
        return category_result
   
   except Exception as e:
        print(e)

   
def calculate_oklid_distance(p_age, p_salary, k, data):
    """
    :param: p_age
    :param: p_salary
    :param: k
    :return:
    """
    try:
        category_1 = {}
        category_0 = {}
        i = 0
        while i < (len(data)) :
            age = data['Age'][i]
            salary = data['Salary'][i]
            purchase_iphone = data['Purchase Iphone'][i]
            if purchase_iphone == 1:
                distance = ((p_age - age)**2 + (p_salary - salary)**2)** 0.5     
                category_1[age, salary] = distance

            else:
                distance = ((p_age - age)**2 + (p_salary - salary)**2)** 0.5       
                category_0[age, salary] = distance

            i += 1
        categories = category_1.copy()
        categories.update(category_0)
        sorted_categories = sorted(categories.items(), key=lambda kv: kv[1])
        return sorted_categories[:k],category_0, category_1
    except Exception as e:
        print(e)
    


def find_category(result, category_0, category_1):
    """
   :param: result
   :return:
   """
    try:
        count_0 = 0
        count_1 = 0
        for i in result:
            for item in category_0.items():
                if  item == i:
                  count_0 += 1
                
            for item in category_1.items():
                if  item == i:
                  count_1 += 1
                
        if count_0 > count_1:
            return 0

        elif count_1 > count_0:
            return 1
        
    except Exception as e:
        print(e)
                
            
def main():
    try:
        data = pd.read_csv("iphone_purchase_records.csv")
        p_age = float(input("Lütfen yaşınızı yazınız: "))
        p_salary = float(input("Lütfen maaşınızı yazınız: "))
        k = int(input("k(en yakın komşu) sayısını giriniz:  "))
        if k < len(data):
           prediction = knn_algorithm((p_age), (p_salary), (k), data )
           print(f"Kategori: {prediction}")
        else:
            print("k değeri dizi boyutundan büyük, tekrar deneyiniz.")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()