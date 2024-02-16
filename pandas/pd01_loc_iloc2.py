import pandas as pd

data = [
    ["Samsung", "1000", "2000"],
     ["Hyundai", "1100", "3000"],
     ["LG", "2000","500"],
     ["Amore","3500", "6000"],
     ["Naver", "100", "1500"]
]

index = ["031", "059", "033", "045", "023"]
columns = ["name", "price", "stock"]

df = pd.DataFrame(data = data, index=index, columns=columns)
print(df)
#         name price stock
# 031  Samsung  1000  2000
# 059  Hyundai  1100  3000
# 033       LG  2000   500
# 045    Amore  3500  6000
# 023    Naver   100  1500
print("===============price 1100원 이상 ======================")
print(df.loc[df["price"] >= "1100"])

print("===============stock 1100원 이상 ======================")
print(df.loc[df["stock"]>= "1100"])

# Convert the "price" column to integers
df["price"] = df["price"].astype(int)

print("===============price 1100원 이상 ======================")
print(df.loc[df["price"] >= 1100, ["name", "stock"]])

aaa = df.loc[df["price"] >= 1100, ["name", "stock"]]
print(aaa)

