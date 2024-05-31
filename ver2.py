import numpy as np
from time import time
from scipy.linalg.blas import sgemm


#Количество строк/столбцов
n = 500

def multiply_matrices_optimized(A, B, block_size):
    # Проверяем размерность матриц
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must be equal to the number of rows in B")
    
    n = len(A)
    C = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Блокирование (Tiling)
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                for i in range(ii, min(ii + block_size, n)):
                    for j in range(jj, min(jj + block_size, n)):
                        temp = C[i][j]
                        for k in range(kk, min(kk + block_size, n)):
                            temp += A[i][k] * B[k][j]
                        C[i][j] = temp
    
    return C

    
#Creating matrix

block_size = 64

matrix1 = np.random.rand(n, n)
matrix2 = np.random.rand(n, n)
matrix1 = np.ascontiguousarray(matrix1)
matrix2 = np.ascontiguousarray(matrix2)


#1 test:
start_time = time()
result_matrix = np.dot(matrix1, matrix2)
end_time = time()

# Вычисляем количество операций с плавающей точкой
num_operations = 2 * n**3
    
# Вычисляем время выполнения в секундах
execution_time = end_time - start_time
    
# Вычисляем производительность в MFlops
mflops = (num_operations / execution_time) / 1e6

print(f"\n======================================================\nОтвет 1 (Использовано {n} строк/столбцов): \n{result_matrix[0][0]}\n")
print(f"Производительность: {mflops:.2f} MFlops")
print(f"Время выполнения: {execution_time:.4f} секунд")

#2 test:
start_time = time()
result_matrix = sgemm(alpha=1.0, a=matrix1, b=matrix2)
end_time = time()

# Вычисляем количество операций с плавающей точкой
num_operations = 2 * n**3
    
# Вычисляем время выполнения в секундах
execution_time = end_time - start_time
    
# Вычисляем производительность в MFlops
mflops = (num_operations / execution_time) / 1e6

print(f"\n======================================================\nОтвет 2 (Использовано {n} строк/столбцов): \n{result_matrix[0][0]}\n")
print(f"Производительность: {mflops:.2f} MFlops")
print(f"Время выполнения: {execution_time:.4f} секунд")

#3 test:
start_time = time()
result_matrix = multiply_matrices_optimized(matrix1, matrix2, block_size)
end_time = time()

# Вычисляем количество операций с плавающей точкой
num_operations = 2 * n**3
    
# Вычисляем время выполнения в секундах
execution_time = end_time - start_time
    
# Вычисляем производительность в MFlops
mflops = (num_operations / execution_time) / 1e6

print(f"\n======================================================\nОтвет 3 (Использовано {n} строк/столбцов): \n{result_matrix[0][0]}\n")
print(f"Производительность: {mflops:.2f} MFlops")
print(f"Время выполнения: {execution_time:.4f} секунд")


