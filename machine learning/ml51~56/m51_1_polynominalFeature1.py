import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(8).reshape(4,2)
print(x)
# #[[0 1]  >0,0,1   -> y값
#  [2 3]  >4,6,9
#  [4 5]  >16,20,25
#  [6 7]]   >36,42,49

pf = PolynomialFeatures(degree=2, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)
# [[ 0.  1.  0.  0.  1.]
#  [ 2.  3.  4.  6.  9.]
#  [ 4.  5. 16. 20. 25.]
#  [ 6.  7. 36. 42. 49.]]

pf = PolynomialFeatures(degree=3, include_bias=False)
x_pf = pf.fit_transform(x)
print(x_pf)
# [[  0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  6.   7.  36.  42.  49. 216. 252. 294. 343.]]

#include_bias=True
# [[  1.   0.   1.   0.   0.   1.   0.   0.   0.   1.]
#  [  1.   2.   3.   4.   6.   9.   8.  12.  18.  27.]
#  [  1.   4.   5.  16.  20.  25.  64.  80. 100. 125.]
#  [  1.   6.   7.  36.  42.  49. 216. 252. 294. 343.]]

print("==================================================")

x = np.arange(8).reshape(4,2)
print(x)
# [[0 1]
#  [2 3]
#  [4 5]
#  [6 7]]
pf = PolynomialFeatures(degree=3, include_bias=True)
x_pf = pf.fit_transform(x)
print(x_pf)
