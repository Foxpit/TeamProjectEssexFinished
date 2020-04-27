#!/usr/bin/env python
# coding: utf-8

# Importing data packages:

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import xgboost
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
from scipy import stats
from scipy.stats import norm, skew
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# In[2]:


dataset = pd.read_csv('train.csv')
dataset_res = pd.read_csv('test.csv')
df = pd.read_csv('train.csv')
y = dataset.iloc[:, -1].values
y_res = dataset_res.iloc[:, -1].values


# In[3]:


dataset.describe()


# In[4]:


dataset_res.describe()


# In[5]:


zero_check_train = dataset.isnull().sum()
zero_check_train = zero_check_train[zero_check_train!=0]
zero_check_train = zero_check_train.sort_values(ascending=True)
zero_check_train


# Corrmat is used for the Data analysis, it shows the relationships between the variables and the strength or the values in regards to one another. This is important as linear regression requires there to be a correlation between the key values and their members.

# In[6]:


corrmat = dataset.corr()
f, ax1 = plt.subplots(figsize=(10,8))
ax1=sns.heatmap(corrmat,vmax = 0.8);


# This shows the correlations between the variables, 1.0 being 100% and 0.00 is 0%. This is why Sales price is 1 as it is going to have a direct correlation as its the same item.

# In[7]:


corr_sale = dataset.corr().SalePrice
corr_field = corr_sale.sort_values(ascending = False).head(20)
corr_field


# Corr_field drops YearRemodAdd GarageCars 1stFlrSF as they do not have an extremely strong correlation and as the prediction models require the highest weighted or similar data, I've dropped the values as it would negatively affect the predicted results

# In[8]:


corr_field = corr_field.drop(['YearRemodAdd','GarageCars','1stFlrSF',]).index


# In[9]:


corrmat = dataset[corr_field].corr()
f, ax1 = plt.subplots(figsize=(12,9))

ax1=sns.heatmap(corrmat,vmax = 0.8,annot = True);


# In[10]:


corr_field = corr_field.drop('SalePrice');


# In[11]:


sns.distplot(y,fit = norm, kde=False);


# This shows the skew of the data within the correlation graph, the notches on the bottom are the individual data points and as the range increased above 400000 the frequency of the data decreases this is due to the dataset being heavy on values between 90000 and 400000.

# In[12]:


sns.distplot(y,fit = norm,rug=True);


# In[13]:


print("Skew: %f" % dataset['SalePrice'].skew(),"Kurt: %f" % dataset['SalePrice'].kurt())


# This Shows the distribution vairance in sales prices along with a line of best fit.

# In[14]:


y_log = np.log(y)
sns.distplot(y_log,fit = norm);


# This is a list of diagrams of each of the dataset and how they impact with the sales price, ive used this to spot outliers and to visualise the variation in the data points and how they weigh against the sales price.

# In[15]:


for i in corr_field:
    plt.scatter(y_log,dataset[i])
    plt.xlabel(i)
    plt.ylabel("sales price")
    plt.show()


# In[ ]:





# In[16]:


X = dataset[corr_field]
X = X.fillna(X.mean())
test_values = dataset_res[corr_field]
test_values = test_values.fillna(test_values.mean())


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import StackingRegressor
from vecstack import stacking
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 42)
KR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
Ela= ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)
XGB = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
Las = Lasso()


ereg = VotingRegressor(estimators=[('gb', XGB), ('KR', KR), ('EN', Ela),('Las', Las)])
ereg = ereg.fit(X_train,y_train)

Las.fit(X_train,y_train)
Ela.fit(X_train,y_train)
XGB.fit(X_train,y_train)
KR.fit(X_train,y_train)

y_pred5 = ereg.predict(X_test)
y_train_pred5 = ereg.predict(X_train)

y_pred = Las.predict(X_test)
y_train_pred = Las.predict(X_train)

y_pred2 = Ela.predict(X_test)
y_train_pred2 = Ela.predict(X_train)

y_pred3 = XGB.predict(X_test)
y_train_pred3 = XGB.predict(X_train)

y_pred4 = KR.predict(X_test)
y_train_pred4 = KR.predict(X_train)


# The final prediction model is a Stacked Regressor taking the best esstimates of each of the regressors and combining the accuracy of the preictive models into one more accurate model.

# In[18]:


from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
estimators = [('ridge', RidgeCV()),('lasso', LassoCV(random_state=42)),('svr', SVR(C=1, gamma=1e-6))]
reg = StackingRegressor(estimators=estimators,final_estimator=GradientBoostingRegressor(random_state=42))
reg.fit(X_train, y_train)

y_pred6 = reg.predict(X_test)
y_train_pred6 = reg.predict(X_train)


# This is a output of the accucy of the predictions

# In[24]:


from sklearn.metrics import r2_score
print("Lasso Train accuracy: " , r2_score(y_train, y_train_pred))
print("Test accuracy: ", r2_score(y_test, y_pred))

print("ElasticNet Train accuracy: " , r2_score(y_train, y_train_pred2))
print("Test accuracy: ", r2_score(y_test, y_pred2))

print("XGBoost Train accuracy: " , r2_score(y_train, y_train_pred3))
print("Test accuracy: ", r2_score(y_test, y_pred3))

print("Kernel Ridge Train accuracy: " , r2_score(y_train, y_train_pred4))
print("Kernel Ridge Test accuracy: ", r2_score(y_test, y_pred4))

print("Combined Voting Regretion Train accuracy: " , r2_score(y_train, y_train_pred5))
print("Combined Voting Regretion Test accuracy: ", r2_score(y_test, y_pred5))

print("stacked Regretion Train accuracy: " , r2_score(y_train, y_train_pred6))
print("Stacked Regretion Test accuracy: ", r2_score(y_test, y_pred6))


# In[20]:


#from sklearn.metrics import mean_squared_error
#print("Train acc: " , clf.score(X_train, y_train))
#print("Test acc: ", clf.score(X_test, y_test))


# In[21]:


final_labels = Las.predict(test_values)
final_labels2 =Ela.predict(test_values)
final_labels3 =XGB.predict(test_values)
final_labels4 =KR.predict(test_values)
final_labels5 = reg.predict(test_values)
final_lables6 = ereg.predict(test_values)

final = [final_labels,final_labels2,final_labels3,final_labels4,final_labels5]


# In[22]:


plt.rcParams["figure.figsize"] = (20,10)
fig, ax = plt.subplots()
fig.suptitle('First 50 value comparison')
ax.plot(final_labels[:50])
ax.plot(final_labels2[:50])
ax.plot(final_labels3[:50])
ax.plot(final_labels4[:50])
ax.plot(final_labels5[:50])


# In[23]:




plt.rcParams["figure.figsize"] = (20,40)
fig2, axs = plt.subplots(6)
axs[1].plot(final_labels[:50])
axs[1].set_title('Laso')
axs[2].plot(final_labels2[:50], 'tab:orange')
axs[2].set_title('ElasticNet')
axs[3].plot(final_labels3[:50], 'tab:green')
axs[3].set_title('XGBoost')
axs[4].plot(final_labels4[:50], 'tab:red')
axs[4].set_title('Ridge')
axs[5].plot(final_labels5[:50], 'tab:green')
axs[5].set_title('Stacked Regretion')
axs[0].plot(final_lables6[:50], 'tab:green')
axs[0].set_title('Combined Voting Regretion')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




