# An outlier is a data point that differs significantly from other similar points. 
# They can occur by chance in any distribution, 
# but they often indicate either measurement error or that the population has a heavy-tailed distribution.

import numpy as np
aaa = np.array([[-10,2,3,4,5,6,7,8,9,10,11,12,50],[100,200,-30,400,500,600,-7000,800,900,1000,210,420,350]]).T

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out > upper_bound) | (data_out < lower_bound))

outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()

# 따릉이, 캐글바이크, 대출, 캐글 비만
#pd04
#이상치를 결측치로 적용한 결과 넣을것