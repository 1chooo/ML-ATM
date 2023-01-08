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


# %%
odd_count(15) # return 7
odd_count(15023) #return 7511

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

# %%
xo('xo'), #True
xo('xo0') #True
xo('xxxoo') #False

# %% [markdown]
# ## Q3. Nth Smallest Element (15%)
# ###Task
# Given an array/list of integers, find the Nth smallest element in the array.
# 
# ###Notes
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
def nth_smallest(arr, pos):
    #your code here


# %%
nth_smallest([3,1,2],2) #ans = 2
nth_smallest([15,20,7,10,4,3],3) #ans = 7 
nth_smallest([-5,-1,-6,-18],4) #and = -1
nth_smallest([-102,-16,-1,-2,-367,-9],5) #ans = -2
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


# %%
square_digits(9119) # ans = 811181
square_digits(0) # ans = 0

# %% [markdown]
# ## Q5. **466920.csv** 為臺北測站 2017 / 01 / 01 ~ 2018 / 11 / 30 之觀測資料，**information.txt** 為欄位說明，將 csv 利用 pandas 讀成 DataFrame 後(5%)完成以下小題。

# %%
import pandas as pd


# %% [markdown]
# ### Q5-1. 計算 2017 年全年均溫及累積降雨量 (5%)

# %%
year_avg_temp = 
year_total_prep = 

# %% [markdown]
# ### Q5-2. 計算 2017年 1~12月各月月均溫及累積降雨量 (5%)

# %%
month_avg_temp = 
month_total_prep = 

# %% [markdown]
# ### Q5-3. 找出 2017 年全年最高溫/最低溫及發生時間 (5%)

# %%
year_max_temp = 
year_max_temp_time = 
year_min_temp = 
year_min_temp_time = 

# %% [markdown]
# ### Q5-4. 找出 2017 年各月最高溫/最低溫 (5%)

# %%
month_max_temp = 
month_min_temp = 

# %% [markdown]
# ### Q5-5. 將 2017 年各月最高溫/最低溫/平均溫以及各月累積降雨量畫在同一張圖中，溫度用折線，雨量用長條，Y軸要分成主副座標分別給溫度及雨量，給分以圖片資訊完整度(標題、軸文字等等)為考量。 (25%)

# %%
%matplotlib inline


