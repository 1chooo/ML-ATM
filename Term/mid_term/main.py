# %% [markdown]
# # 天氣與人工智慧 期中考
# 上傳時檔名記得將學號與姓名改掉
# 
# 考試時間至 11/06 (Sun.) **23:59** 為止
# 
# 請根據題目敘述完成自訂函數
# 
# 各題皆有提供測試函數的 cell，供各位驗證自己的函數是否正確
# 
# 但是請注意驗證 cell 的答案答對不一定代表你的函數沒問題
# 
# 還是盡量要寫得通用一點
# 
# 祝各位作答愉快

# %% [markdown]
# ## Q1. Count Odd Numbers below n (10%)
# 
# Given a number n, return the number of positive odd numbers below n.
# 
# Examples (Input -> Output)
# 
# 7  -> 3 (because odd numbers below 7 are [1, 3, 5])
# 
# 15 -> 7 (because odd numbers below 15 are [1, 3, 5, 7, 9, 11, 13])

# %%
def odd_count(n):

    #your code here
    
    count = 0

    for i in range(1, n) :
        if (i % 2 != 0) :
            count += 1
        
    return count

# %%
odd_count(15) # return 7
odd_count(15023) #return 7511

# %% [markdown]
# ### Test 1

# %%
odd_count(15)

# %% [markdown]
# ### Test 2

# %%
odd_count(15023)

# %% [markdown]
# ## Q2. Exes and Ohs (10%)
# Check to see if a string has the same amount of 'x's and 'o's. 
# 
# The method must return a boolean and be case insensitive. 
# 
# The string can contain any char.
# 
# Examples input/output:
# 
# XO("ooxx") => true
# 
# XO("xooxx") => false
# 
# XO("ooxXm") => true
# 
# XO("zpzpzpp") => true // when no 'x' and 'o' is present should return true
# 
# XO("zzoo") => false
# 
# 

# %%
def xo(s):
    
    #your code here

    count_x = 0
    count_o = 0

    for i in range(0, len(s)) :
        if (s[i] == 'x' or s[i] == 'X') :
            count_x += 1
        
        if (s[i] == 'o' or s[i] == 'O') :
            count_o += 1

    if (count_x == count_o) :
        return True

    return False

# %%
print(xo("ooxx"))
print(xo("xooxx"))
print(xo("ooxXm"))
print(xo("zpzpzpp"))
print(xo("zzoo"))

# %%
xo('xo'), #True
xo('xo0') #True
xo('xxxoo') #False

# %% [markdown]
# ### Test 1

# %%
xo('xo') #True

# %% [markdown]
# ### Test 2

# %%
xo('xo0') #True

# %% [markdown]
# ### Test 3

# %%
xo('xxxoo') #False

# %% [markdown]
# ## Q3. Nth Smallest Element (15%)
# ### Task
# Given an array/list of integers, find the Nth smallest element in the array.
# 
# ### Notes
# Array/list size is at least 3.
# 
# Array/list's numbers could be a mixture of positives , negatives and zeros.
# 
# Repetition in array/list's numbers could occur, so don't remove duplications.
# 
# Input >> Output Examples
# 
# arr=[3,1,2]            n=2    ==> return 2 
# 
# arr=[15,20,7,10,4,3]   n=3    ==> return 7 
# 
# arr=[2,169,13,-5,0,-1] n=4    ==> return 2 
# 
# arr=[2,1,3,3,1,2],     n=3    ==> return 2 

# %%
def quick_sort(arr, lb, rb) :
    
    if (lb >= rb) :
        return arr

    pivot = arr[rb]
    l = lb
    r = rb - 1

    while True :
        while (arr[l] < pivot) :
            l += 1
        
        while (arr[r] >= pivot and r > lb) :
            r -= 1

        if (l < r) :
            temp = arr[l]
            arr[l] = arr[r]
            arr[r] = temp
        else :
            break

    if (arr[rb] != arr[l]) :
        temp = arr[rb]
        arr[rb] = arr[l]
        arr[l] = temp

    quick_sort(arr, lb, l - 1)
    quick_sort(arr, l + 1, rb)
    
    return arr

def nth_smallest(arr, pos):
    #your code here

    arr_len = len(arr)
    quick_sort(arr, 0, arr_len - 1)
    # print(arr)
    
    return arr[pos - 1]

# %%
nth_smallest([3,1,2],2) #ans = 2
nth_smallest([15,20,7,10,4,3],3) #ans = 7 
nth_smallest([-5,-1,-6,-18],4) #and = -1
nth_smallest([-102,-16,-1,-2,-367,-9],5) #ans = -2
nth_smallest([2,169,13,-5,0,-1],4) #ans = 2

# %% [markdown]
# ### Test 1

# %%
nth_smallest([3,1,2],2)

# %% [markdown]
# ### Test 2

# %%
nth_smallest([15,20,7,10,4,3],3) #ans = 7 

# %% [markdown]
# ### Test 3

# %%
nth_smallest([-5,-1,-6,-18],4) #and = -1

# %% [markdown]
# ### Test 4
# 

# %%
nth_smallest([-102,-16,-1,-2,-367,-9],5) #ans = -2

# %% [markdown]
# ### Test 5

# %%
nth_smallest([2,169,13,-5,0,-1],4) #ans = 2

# %% [markdown]
# ## Q4. Square Every Digit (15%)
# you are asked to square every digit of a number and concatenate them.
# 
# For example, if we run 9119 through the function, 811181 will come out, because 9^2 is 81 and 1^2 is 1.
# 
# Note: The function accepts an integer and returns an integer

# %%
def square_digits(num):
    # your code

    ans = ""

    num = str(num)

    for i in range(0, len(num)) :
        temp = int(num[i])
        temp = temp * temp
        temp = str(temp)
        ans += temp
    
    ans = int(ans)
    
    return ans

# %%
square_digits(9119) # ans = 811181
square_digits(0) # ans = 0

# %% [markdown]
# ### Test 1

# %%
square_digits(9119) # ans = 811181

# %% [markdown]
# ### Test 2

# %%
square_digits(0) # ans = 0

# %% [markdown]
# ## Q5. **466920.csv** 為臺北測站 2017 / 01 / 01 ~ 2018 / 11 / 30 之觀測資料，**information.txt** 為欄位說明，將 csv 利用 pandas 讀成 DataFrame 後(5%)完成以下小題。

# %%
import pandas as pd

# %%
data_frame = pd.read_csv("./466920.csv")

# %%
data_frame.info()

# %%
data_frame.head()

# %%
data_frame = data_frame.replace(-9991, 0)
data_frame = data_frame.replace(-9996, 0)
data_frame = data_frame.replace(-9997, 0)
data_frame = data_frame.replace(-9998, 0)
data_frame = data_frame.replace(-9999, 0)

# %% [markdown]
# ### Q5-1. 計算 2017 年全年均溫及累積降雨量 (5%)

# %%


# %%
year_avg_temp = float(data_frame[data_frame["year"] == 2017][["TX01"]].mean())
year_total_prep = float(data_frame[data_frame["year"] == 2017][["PP01"]].sum())

print("Average temperature in 2017: ", year_avg_temp, "°C")
print()
print("Total accumulate precipitation in 2017: ", year_total_prep, "mm")

# %% [markdown]
# ### Q5-2. 計算 2017年 1~12月各月月均溫及累積降雨量 (5%)

# %%
month_avg_temp_list = []
month_total_prep_list = []
month_list = ["January  ", "February ", "March    ",
              "April    ", "May      ", "June     ",
              "July     ", "August   ", "September",
              "Octoer   ", "November ", "December "]

temp_data_frame = data_frame[data_frame["year"] == 2017]

for i in range(1, 13) :
    temp = temp_data_frame[temp_data_frame["month"] == i][["TX01"]].mean()
    temp = float(temp)
    month_avg_temp_list.append(temp)

    temp = temp_data_frame[temp_data_frame["month"] == i][["PP01"]].sum()
    temp = float(temp)
    month_total_prep_list.append(temp)

for i in range(0, 12) :
    print(f"In 2017, {month_list[i]}: \tMean monthly temperature: {month_avg_temp_list[i]} °C; \tTotal accumulate precipitation: {month_total_prep_list[i]}")


# month_avg_temp = 
# month_total_prep = 

# %% [markdown]
# ### Q5-3. 找出 2017 年全年最高溫/最低溫及發生時間 (5%)

# %%
# temp_data_frame = data_frame[data_frame["year"] == 2017]

year_max_temp = float(data_frame[data_frame["year"] == 2017][["TX01"]].max())
year_max_temp_time = data_frame[data_frame["TX01"] == year_max_temp][["yyyymmddhh"]]
year_min_temp = float(data_frame[data_frame["year"] == 2017][["TX01"]].min())
year_min_temp_time = data_frame[data_frame["TX01"] == year_min_temp][["yyyymmddhh"]]

print("The maximum temperature in 2017,", year_max_temp, "°C and it happens on: ")
print(year_max_temp_time)
print()
print("The minimum temperature in 2017,", year_min_temp, "°C and it happens on: ")
print(year_min_temp_time)

# %% [markdown]
# ### Q5-4. 找出 2017 年各月最高溫/最低溫 (5%)

# %%
month_max_temp_list = []
month_min_temp_list = []
temp_data_frame = data_frame[data_frame["year"] == 2017]

for i in range(1, 13) :
    temp = temp_data_frame[temp_data_frame["month"] == i][["TX01"]].max()
    temp = float(temp)
    month_max_temp_list.append(temp)

    temp = temp_data_frame[temp_data_frame["month"] == i][["TX01"]].min()
    temp = float(temp)
    month_min_temp_list.append(temp)


for i in range(0, 12) :
    print(f"In 2017, {month_list[i]}: \tMaximum temperature: {month_max_temp_list[i]} °C; \tMinimum precipitation: {month_min_temp_list[i]} °C")

# month_max_temp = 
# month_min_temp = 

# %% [markdown]
# ### Q5-5. 將 2017 年各月最高溫/最低溫/平均溫以及各月累積降雨量畫在同一張圖中，溫度用折線，雨量用長條，Y軸要分成主副座標分別給溫度及雨量，給分以圖片資訊完整度(標題、軸文字等等)為考量。 (25%)

# %%
%matplotlib inline

import matplotlib.pyplot as plt

x_month = [*range(1, 13, 1)]

fig, ax1 = plt.subplots()

ax1.plot(x_month, month_avg_temp_list, "k-o")
ax1.plot(x_month, month_max_temp_list, "r-o")
ax1.plot(x_month, month_min_temp_list, "g-o")

ax1.set_ylabel("°C", rotation=0)


ax2 = ax1.twinx()
ax2.bar(x_month, month_total_prep_list, alpha=0.5)

ax2.set_ylabel("mm", rotation=0)

ax1.legend(["avg temp", "max temp", "min temp"], loc ="upper left")
ax2.legend(["total prep"], loc="upper right")

plt.xlim(0, 13)
plt.grid()
plt.xlabel("Month")
plt.title("Total precipitation and Temperature diversity in 2017 by Ho")

plt.savefig("./result.jpg", dpi=300)
plt.show()

# %% [markdown]
# ## Add my personal icon as watermark
# 
# ![plot](./src/icon.png)

# %%
import os
import glob
from PIL import Image


icon = Image.open('./src/icon.png')
img = Image.open("./result.jpg")

img.paste(icon, (0, 0), icon)   
img.save(f'./watermark/result.jpg')


