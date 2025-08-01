import pandas as pd 

data = pd.read_csv("iphone_purchase_records.csv")
purchase_iphone = data['Purchase Iphone']

def knn_algorithm(p_age, p_salary, k):
   """
   :param: p_age
   :param: p_salary
   :param: k
   :return:
   """
   result = calculate_oklid_distance( p_age, p_salary, k)
   category_result = find_category(result)
   return category_result

category_1 = {}
category_0 = {}
def calculate_oklid_distance(p_age, p_salary, k):
    """
    :param: p_age
    :param: p_salary
    :param: k
    :return:
    """
    i = 0
    while i < (len(data['Age'])) :
        age = data['Age'][i]
        salary = data['Salary'][i]

        if purchase_iphone[i] == 1:
            distance = ((p_age - age)**2 + (p_salary - salary)**2)** 0.5     
            category_1[(float(age), float(salary))] = (float(distance))

        else:
            distance = ((p_age - age)**2 + (p_salary - salary)**2)** 0.5       
            category_0[(float(age), float(salary))] = (float(distance))

        i += 1
    categories = category_1.copy()
    categories.update(category_0)
    sorted_categories = sorted(categories.items(), key=lambda kv: kv[1])
    return sorted_categories[:k]
    
def find_category(result):
    """
   :param: result
   :return:
   """
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
            
            
def main():
    try:
        p_age = float(input("Lütfen yaşınızı yazınız: "))
        p_salary = float(input("Lütfen maaşınızı yazınız: "))
        k = int(input("k(en yakın komşu) sayısını giriniz:  "))
        prediction = knn_algorithm((p_age), (p_salary), (k))
        print(f"Kategori: {prediction}")
    except:
        print("Lütfen yanıtlarınızı kontrol edip tekrar deneyiniz")

if __name__ == "__main__":
    main()