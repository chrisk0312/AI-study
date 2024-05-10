#4-2 python 버블정렬 알고리즘
import random
import time

def Bubble_Sort(A):
    n = len(A)
    for i in range(n-1):
        sorted = True
        for j in range(0, n-i-1):
            if A[j] > A[j+1]:
                A[j], A[j+1] = A[j+1], A[j]
                sorted = False
        if sorted:
            break
    return A

A = [random.randint(1, 100) for _ in range(100)]
print(f"정렬: {A}")
start_time = time.time()
sorted = Bubble_Sort(A)
end_time = time.time()
time = end_time - start_time
print(f"버블정렬: {sorted}")
print(f"소요시간: {time}초")
