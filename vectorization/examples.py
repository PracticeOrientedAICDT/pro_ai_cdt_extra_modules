import numpy as np 
import scipy.linalg as la
import time

N = 200
A = np.random.rand(N, N)
B = np.random.rand(N, N)

def dot_product_comparison():

    start = time.perf_counter()
    C_loop = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            s = 0.0
            for k in range(N):
                s += A[i, k] * B[k, j]
            C_loop[i, j] = s

    t_loop = time.perf_counter() - start

    start = time.perf_counter()
    C_vec = np.dot(A, B)
    t_vec = time.perf_counter() - start

    print('Dot Product\n')
    print(f"Loop: {t_loop} s")
    print(f"Vectorized: {t_vec} s")
    print(f"Speedup:     {t_loop / t_vec:.0f}x")
    print(np.allclose(C_loop, C_vec, atol=1e-10))
    print('\n')

def lu_decompose(A):
    """Returns P, L, U such as  P @ A = L @ U."""
    A = A.astype(float).copy()
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))

    for k in range(n):
       
        piv = k + np.argmax(np.abs(A[k:, k]))
        if piv != k:
            A[[k, piv]] = A[[piv, k]]      
            P[[k, piv]] = P[[piv, k]]     
            L[[k, piv], :k] = L[[piv, k], :k] 

        if A[k, k] == 0:
            raise ValueError("Matriz singular")

        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            L[i, k] = m
            A[i, k:] -= m * A[k, k:]

    L += np.eye(n) 
    U = np.triu(A)
    return P, L, U
        
def lu_comparision():

    N = 200
    A = np.random.rand(N, N)

    start = time.perf_counter()
    P_loop, L_loop, U_loop = lu_decompose(A)
    t_loop = time.perf_counter() - start

    start = time.perf_counter()
    P, L, U_lu = la.lu(A)
    t_vec = time.perf_counter() - start

    print('LU Decomposition\n')
    print(f'Vectorized: {t_vec:.2f} s')
    print(f'Loop: {t_loop:.2f} s')
    print(f'Speedup: {(t_loop / t_vec):.2f} x')


if __name__ == '__main__':

    dot_product_comparison()

    lu_comparision()
