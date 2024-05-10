#4-2 phython 삽입정렬 알고리즘
import random
import time

def Insertion_Sort(A):
    n = len(A)
    for i in range(1, n):
        val = A[i]
        j = i
        while j > 0 and A[j - 1] > val:
            A[j] = A[j - 1]
            j -= 1
        A[j] = val
    return A

A = [random.randint(1, 100) for _ in range(100)]
print(f"정렬: {A}")
start_time = time.time()
sorted = Insertion_Sort(A)
end_time = time.time()
time = end_time - start_time
print(f"선택정렬: {sorted}")
print(f"소요시간: {time}초")

