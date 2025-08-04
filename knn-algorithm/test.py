import pandas as pd
import time
import sys
import os

from knn_without_library import knn_algorithm
from knn_with_library import sklearn_model_predict

def run_test():
    """
    Kütüphane kullanılmadan ve sklearn kütüphanesi kullanılarak yazılan iki KNN algoritmasının tahmin doğruluğu ve çalışma süresini karşılaştırır.
    """
    data = pd.read_csv("iphone_purchase_records.csv")
    feature_cols = list(data.columns[:-1])
    label_col = data.columns[-1]
    k = 3

    test_input = [32, 160000]
    age, salary = test_input

    start_custom = time.time()
    result_custom = knn_algorithm(feature_cols, label_col, data, test_input, k)
    end_custom = time.time()
    duration_custom = end_custom - start_custom

    start_sklearn = time.time()
    result_sklearn = sklearn_model_predict(age, salary, k, csv_path="iphone_purchase_records.csv")
    end_sklearn = time.time()
    duration_sklearn = end_sklearn - start_sklearn

    print("KNN Sonuç Karşılaştırma Raporu")
    print(f"Test Girdisi: {test_input} | k = {k}\n")
    print(f"Kütüphane kullanılmadan KNN Tahmini : {result_custom} | Süre: {duration_custom:.6f} sn")
    print(f"sklearn Kütüphanesi kullanılarak KNN Tahmini : {result_sklearn} | Süre: {duration_sklearn:.6f} sn")

if __name__ == "__main__":
    run_test()
