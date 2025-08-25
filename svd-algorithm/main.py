import sympy as sp
import numpy as np


def matrix_multiply(A, B):
    """
    İki matrisi çarpar.
    A: m x n matrisi
    B: n x p matrisi
    return: m x p boyutunda çarpım matrisi
    """
    C = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
    return C


def matrix_vector_multiply(A, v):
    """
    Matris A ile vektör v çarpımı.
    return: sonuç vektör
    """
    C = [0 for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(v)):
            C[i] += A[i][j] * v[j]
    return C


def transpose_matrix(A):
    """
    A matrisinin transpozunu alır ve A^T*A hesaplar.
    return: kare matris (A^T * A)
    """
    return A.T @ A


def compute_eigenvalues(A_t):
    """
    Kare matrisin özdeğerlerini bulur.
    Sympy ile determinant denklemi çözülür:
    return: özdeğerler listesi
    """
    x = sp.symbols('x') 
    equation = sp.Eq(sp.det(A_t - x*sp.eye(A_t.shape[0])), 0)
    solutions = sp.solve(equation, x)
    return solutions


def compute_eigenvectors(eigenvals, A_t):
    """
    Her özdeğer için (A - λI)v = 0 denkleminden özvektörleri bulur.
    Nullspace kullanılır ve birim vektör olarak normalize edilir.
    return: özvektör listesi (V matrisi)
    """
    singular_vectors = []
    for val in eigenvals:
        null_space = (A_t - val*sp.eye(A_t.shape[0])).nullspace()
        for v in null_space:
            norm = (sum([x**2 for x in v]))**0.5
            v_normalized = v / norm
            singular_vectors.append([float(x) for x in v_normalized])
    return singular_vectors


def compute_singular_values(eigenvals):
    """
    Özdeğerlerden singular değerleri hesaplar.
    tek boyutlu liste döndürür.
    """
    return sorted([float(val**0.5) for val in eigenvals])


def compute_left_singular_vectors(A, V, sigma_values):
    """
    U matrisini hesaplar.
    Her sigma ve v için: u = A*v / sigma
    Sonra normalize edilir.
    """
    U_cols = []
    for i in range(len(sigma_values)):
        sigma = sigma_values[i]
        v = [row[i] for row in V]
        C = matrix_vector_multiply(A, v)
        u = [val / sigma for val in C]
        U_cols.append(u)
    return horizontal_stack(U_cols)


def horizontal_stack(columns):
    return [[columns[j][i] for j in range(len(columns))] for i in range(len(columns[0]))]


def main():
    
    A = np.array([
        [4, 4],
        [3, -3]
    ])
    
    
    A_t = transpose_matrix(A)
    
 
    eigvals = compute_eigenvalues(A_t)
    

    V = compute_eigenvectors(eigvals, A_t)
    
    sigma_values = compute_singular_values(eigvals)
    
    U = compute_left_singular_vectors(A, V, sigma_values)

    A_reconstructed = [[0]*len(A[0]) for _ in range(len(A))]
    for k in range(len(sigma_values)):
        for i in range(len(A)):
            for j in range(len(A[0])):
                A_reconstructed[i][j] += sigma_values[k] * U[i][k] * V[j][k]

    k = 1
    A_compressed = [[0]*len(A[0]) for _ in range(len(A))]
    for idx in range(k):
        for i in range(len(A)):
            for j in range(len(A[0])):
                A_compressed[i][j] += sigma_values[idx] * U[i][idx] * V[j][idx]

    
    print("\n--- Orijinal Matris A ---")
    print(A)
    print("\n--- A matrisi yeniden çarpımı ---")
    print(A_reconstructed)
    print(f"\n--- Sıkıştırılmış Matris (k={k}) ---")
    print(A_compressed)

main()
