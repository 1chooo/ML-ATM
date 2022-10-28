# %% [markdown]
# # 一、機器學習評估指標選定
# ## [教學目標]
# 學習 sklearn 中，各種評估指標的使用與意義

# %% [markdown]
# ## [範例重點]
# 注意觀察各指標的數值範圍，以及輸入函數中的資料格式

# %%
from sklearn import metrics, datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np
%matplotlib inline

# %% [markdown]
# ## 回歸問題
# 常見的評估指標有
# - MAE
# - MSE
# - R-square

# %% [markdown]
# 我們隨機生成(X, y)資料，然後使用線性回歸模型做預測，再使用 MAE, MSE, R-square 評估

# %%
X, y = datasets.make_regression(n_features=1, random_state=42, noise=100) # 生成資料
model = LinearRegression() # 建立回歸模型
model.fit(X, y) # 將資料放進模型訓練
prediction = model.predict(X) # 進行預測
mae = metrics.mean_absolute_error(prediction, y) # 使用 MAE 評估
mse = metrics.mean_squared_error(prediction, y) # 使用 MSE 評估
r2 = metrics.r2_score(prediction, y) # 使用 r-square 評估
print("MAE: ", mae)
print("MSE: ", mse)
print("R-square: ", r2)

# %%
plt.scatter(X,y)
plt.show()

# %%
plt.scatter(X, prediction)
plt.show()

# %% [markdown]
# ## 分類問題
# 常見的評估指標有
# - AUC
# - F1-Score (Precision, Recall)

# %%
cancer = datasets.load_breast_cancer() # 我們使用 sklearn 內含的乳癌資料集
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=50, random_state=0)

# %%
print(y_test) # 測試集中的 label

# %%
print(X_train)

# %%
y_pred = np.random.random((50,)) # 我們先隨機生成 50 筆預測值，範圍都在 0~1 之間，代表機率值

# %%
print(y_pred)

# %% [markdown]
# ### AUC

# %%
auc = metrics.roc_auc_score(y_test, y_pred) # 使用 roc_auc_score 來評估。 **這邊特別注意 y_pred 必須要放機率值進去!**
print("AUC: ", auc) # 得到結果約 0.5，與亂猜的結果相近，因為我們的預測值是用隨機生成的

# %% [markdown]
# ## F1-Score

# %%
threshold = 0.5 
y_pred_binarized = np.where(y_pred>threshold, 1, 0) # 使用 np.where 函數, 將 y_pred > 0.5 的值變為 1，小於 0.5 的為 0
f1 = metrics.f1_score(y_test, y_pred_binarized) # 使用 F1-Score 評估
precision = metrics.precision_score(y_test, y_pred_binarized) # 使用 Precision 評估
recall  = metrics.recall_score(y_test, y_pred_binarized) # 使用 recall 評估
print("F1-Score: ", f1) 
print("Precision: ", precision)
print("Recall: ", recall)

# %% [markdown]
# ## [本節重點]
# 了解 F1-score 的公式意義，並試著理解程式碼

# %% [markdown]
# ## 練習
# 請參考 F1-score 的公式與[原始碼](https://github.com/scikit-learn/scikit-learn/blob/bac89c2/sklearn/metrics/classification.py#L620)，試著寫出 F2-Score 的計算函數

# %%


# %% [markdown]
# ---

# %% [markdown]
# # 二、Regression 模型
# ## [教學重點]
# 學習使用 sklearn 中的 linear regression 模型，並理解各項參數的意義

# %% [markdown]
# ## [範例重點]
# 觀察丟進模型訓練的資料格式，輸入 linear regression 與 Logistic regression 的資料有甚麼不同?

# %% [markdown]
# ## import 需要的套件

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# %% [markdown]
# ### Linear regression

# %%
# 讀取糖尿病資料集
diabetes = datasets.load_diabetes()

# 為方便視覺化，我們只使用資料集中的 1 個 feature (column)
X = diabetes.data[:, np.newaxis, 2]
print("Data shape: ", X.shape) # 可以看見有 442 筆資料與我們取出的其中一個 feature

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(X, diabetes.target, test_size=0.1, random_state=4)

# 建立一個線性回歸模型
regr = linear_model.LinearRegression()

# 將訓練資料丟進去模型訓練
regr.fit(x_train, y_train)

# 將測試資料丟進模型得到預測結果
y_pred = regr.predict(x_test)

# %%
diabetes.data.shape

# %%
# 可以看回歸模型的參數值
print('Coefficients: ', regr.coef_)

# 預測值與實際值的差距，使用 MSE
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# %%
# 畫出回歸模型與實際資料的分佈
plt.scatter(x_test, y_test,  color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.show()

# %% [markdown]
# ### Logistics regression

# %%
# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1, random_state=4)

# 建立模型
logreg = linear_model.LogisticRegression()

# 訓練模型
logreg.fit(x_train, y_train)

# 預測測試集
y_pred = logreg.predict(x_test)

# %%
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# %% [markdown]
# ## [練習重點]
# 了解其他資料集的使用方法，如何將資料正確地送進模型訓練

# %% [markdown]
# ## 練習時間
# 試著使用 sklearn datasets 的其他資料集 (wine, boston, ...)，來訓練自己的線性迴歸模型。

# %%
wine = datasets.load_wine()
boston = datasets.load_boston()
breast_cancer = datasets.load_breast_cancer()

# %% [markdown]
# ### HINT: 注意 label 的型態，確定資料集的目標是分類還是回歸，再使用正確的模型訓練！

# %% [markdown]
# ---

# %% [markdown]
# # 三、Lasso、Ridge Regression 模型

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# %%
# 讀取糖尿病資料集
diabetes = datasets.load_diabetes()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=4)

# 建立一個線性回歸模型
regr = linear_model.LinearRegression()

# 將訓練資料丟進去模型訓練
regr.fit(x_train, y_train)

# 將測試資料丟進模型得到預測結果
y_pred = regr.predict(x_test)

# %%
x_train.shape

# %%
x_train[1]

# %%
y = x1*w1 + x2*w2 + .... + x10*w10 + b

# %%
print(regr.coef_)

# %%
# 預測值與實際值的差距，使用 MSE
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# %% [markdown]
# ### LASSO

# %%
# 讀取糖尿病資料集
diabetes = datasets.load_diabetes()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=4)

# 建立一個線性回歸模型
lasso = linear_model.Lasso(alpha=1.0)

# 將訓練資料丟進去模型訓練
lasso.fit(x_train, y_train)

# 將測試資料丟進模型得到預測結果
y_pred = lasso.predict(x_test)

# %%
# 印出各特徵對應的係數，可以看到許多係數都變成 0，Lasso Regression 的確可以做特徵選取
lasso.coef_

# %%
# 預測值與實際值的差距，使用 MSE
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# %% [markdown]
# ### Ridge

# %%
# 讀取糖尿病資料集
diabetes = datasets.load_diabetes()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=4)

# 建立一個線性回歸模型
ridge = linear_model.Ridge(alpha=10)

# 將訓練資料丟進去模型訓練
ridge.fit(x_train, y_train)

# 將測試資料丟進模型得到預測結果
y_pred = regr.predict(x_test)

# %%
# 印出 Ridge 的參數，可以很明顯看到比起 Linear Regression，參數的數值都明顯小了許多
print(ridge.coef_)

# %%
# 預測值與實際值的差距，使用 MSE
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))

# %% [markdown]
# 可以看見 LASSO 與 Ridge 的結果並沒有比原本的線性回歸來得好，
# 這是因為目標函數被加上了正規化函數，讓模型不能過於複雜，相當於限制模型擬和資料的能力。因此若沒有發現 Over-fitting 的情況，是可以不需要一開始就加上太強的正規化的。

# %% [markdown]
# ## 練習時間

# %% [markdown]
# 請使用其他資料集 (boston, wine)，並調整不同的 alpha 來觀察模型訓練的情形。

# %%


# %% [markdown]
# # 四、決策樹
# ## [範例重點]
# 了解機器學習建模的步驟、資料型態以及評估結果等流程

# %%
from sklearn import datasets, metrics

# 如果是分類問題，請使用 DecisionTreeClassifier，若為回歸問題，請使用 DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## 建立模型四步驟
# 
# 在 Scikit-learn 中，建立一個機器學習的模型其實非常簡單，流程大略是以下四個步驟
# 
# 1. 讀進資料，並檢查資料的 shape (有多少 samples (rows), 多少 features (columns)，label 的型態是什麼？)
#     - 讀取資料的方法：
#         - **使用 pandas 讀取 .csv 檔：**pd.read_csv
#         - **使用 numpy 讀取 .txt 檔：**np.loadtxt 
#         - **使用 Scikit-learn 內建的資料集：**sklearn.datasets.load_xxx
#     - **檢查資料數量：**data.shape (data should be np.array or dataframe)
# 2. 將資料切為訓練 (train) / 測試 (test)
#     - train_test_split(data)
# 3. 建立模型，將資料 fit 進模型開始訓練
#     - clf = DecisionTreeClassifier()
#     - clf.fit(x_train, y_train)
# 4. 將測試資料 (features) 放進訓練好的模型中，得到 prediction，與測試資料的 label (y_test) 做評估
#     - clf.predict(x_test)
#     - accuracy_score(y_test, y_pred)
#     - f1_score(y_test, y_pred)

# %%
# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型
clf = DecisionTreeClassifier()

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

# %%
acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)

# %%
print(iris.feature_names)

# %%
print("Feature importance: ", clf.feature_importances_)

# %% [markdown]
# ## 練習
# 
# 1. 試著調整 DecisionTreeClassifier(...) 中的參數，並觀察是否會改變結果？
# 2. 改用其他資料集 (boston, wine)，並與回歸模型的結果進行比較

# %%


# %% [markdown]
# # 五、隨機森林
# ## [範例重點]
# 了解隨機森林的建模方法及其中超參數的意義

# %%
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# %%
# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型 (使用 20 顆樹，每棵樹的最大深度為 4)
clf = RandomForestClassifier(n_estimators=1000, max_depth=10)

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

# %%
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

# %%
print(iris.feature_names)

# %%
print("Feature importance: ", clf.feature_importances_)

# %% [markdown]
# ## 練習
# 
# 1. 試著調整 RandomForestClassifier(...) 中的參數，並觀察是否會改變結果？
# 2. 改用其他資料集 (boston, wine)，並與回歸模型與決策樹的結果進行比較

# %%


# %% [markdown]
# ---

# %% [markdown]
# # 六、梯度提升機

# %%
from sklearn import datasets, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# %%
# 讀取鳶尾花資料集
iris = datasets.load_iris()

# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=4)

# 建立模型
clf = GradientBoostingClassifier()

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

# %%
acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)

# %% [markdown]
# ## 練習
# 目前已經學過許多的模型，相信大家對整體流程應該比較掌握了，這次練習請改用**手寫辨識資料集**，步驟流程都是一樣的，請試著自己撰寫程式碼來完成所有步驟

# %%
digits = datasets.load_digits()

# %%
!python --version

# %% [markdown]
# ---


