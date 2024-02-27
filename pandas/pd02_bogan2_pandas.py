import numpy as np
import pandas as pd

data = pd.DataFrame([[2, np.nan, 6, 8, 10],
                     [2,4,np.nan,8,np.nan],
                     [2,4,6,8,10],
                     [np.nan,4,np.nan,8,np.nan]])

data = data.transpose()
data.columns = ["x1", "x2", "x3", "x4"]
print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 결측치 확인
print(data.isnull())
print(data.isnull().sum())
print(data.info())

# 결측치 처리
# print(data.dropna()) #default axis=0
# print(data.dropna(axis=0)) #행 삭제
print(data.dropna(axis=1)) #열 삭제

#2-1.  mean
means = data.mean()
print(means)
data2 = data.fillna(means)
print(data2)

#2-2. median
medians = data.median()
print(medians)
data3 = data.fillna(medians)
print(data3)

# 2-3. 특정값 0채우기
data4 = data.fillna(0)
print(data4)

data4_2 = data.fillna(777)
print(data4_2)

# 2-4. frontfill으로 채우기
data5 = data.fillna(method="ffill")
print(data5)

#2-5. backfill로 채우기
data6 = data.bfill()
print(data6)

means = data['x1'].mean()
print(means)

median = data['x4'].median()
print(median)




data["x1"] = data["x1"].fillna(means)
data["x4"] = data["x4"].fillna(median)


print(data)

