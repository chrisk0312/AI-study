#4-2 phython 선택정렬 알고리즘
import random
import time

def Selection_Sort(A):
    n = len(A)
    for i in range(n-1):  
        min_idx = i
        for j in range(i+1, n):  
            if A[min_idx] > A[j]:
                min_idx = j
        A[i], A[min_idx] = A[min_idx], A[i]
    return A

A = [random.randint(1, 100) for _ in range(100)]
print(f"정렬: {A}")
start_time = time.time()
sorted = Selection_Sort(A)
end_time = time.time()
time = end_time - start_time
print(f"선택정렬: {sorted}")
print(f"소요시간: {time}초")

