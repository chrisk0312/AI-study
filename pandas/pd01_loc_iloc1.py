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
print("=====================================")
# print(df[0])#error
# print(df["031"])#error
print(df["name"]) #index가 column이 되어버림, pandas에서는 column 기준
print(df["name"],["045"])#pands  열행 기준

# loc : index 기준으로 행 데이터 추출
# iloc : columns 기준으로 행 데이터 추출
# 인트 형태로 인덱스를 주면 loc는 행번호로 인식

#         name price stock
# 031  Samsung  1000  2000
# 059  Hyundai  1100  3000
# 033       LG  2000   500
# 045    Amore  3500  6000
# 023    Naver   100  1500

print("=====================================")
print(df.loc["045"]) #행 데이터 추출
print("=====================================")
# print(df.loc[3]) #행 데이터 추출 #error
print("=====================================")
print(df.iloc[3]) #(df.loc["045"]) 동일 결과 
print("=====================================")
print(df.loc["023"]) 
print(df.iloc[4]) 
print(df.iloc[-1]) 
print("=====================================")
print(df.loc["045"].loc["price"]) #3500
print(df.loc["045"].iloc[1]) #3500
print(df.iloc[3].iloc[1]) #3500
print(df.iloc[3].loc["price"]) #3500

print(df.loc["045"][1]) #3500
print(df.iloc[3][1]) #3500

print(df.loc["045"]["price"]) #3500
print(df.iloc[3]["price"]) #3500

print(df.loc["045", "price"]) #3500
print(df.iloc[3,1]) #3500

print("=====================================")
print(df.iloc[3:5, 1])
print(df.iloc[[3,4], 1])

print(df.loc["045":"023", "price"]) #error
print(df.loc[["045","023"], "price"]) #error
