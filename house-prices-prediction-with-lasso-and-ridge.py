#!/usr/bin/env python
# coding: utf-8

# In[152]:


##Importing necessary Libraries
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import math
import matplotlib.pyplot as plt
from scipy.stats import skew
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


#pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


# In[153]:


##Loading Data
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train_df.head()


# In[154]:


# Describe the Datasets
#train_df.shape, test_df.shape
#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train_df.shape))
print("The test data size before dropping Id feature is : {} ".format(test_df.shape))

#Save the 'Id' column
train_ID = train_df['Id']
test_ID = test_df['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train_df.drop("Id", axis = 1, inplace = True)
test_df.drop("Id", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train_df.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test_df.shape))


# Train has 1460 rows and 81 columns
# Test has 1459 rows and 80 columns

# # Outliers

# In[155]:


fig, ax = plt.subplots()
ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# We can see at the rightmost two with extremely large GrLivArea that are of a low price. These values are huge oultliers. Therefore, we can safely delete them.

# In[156]:


#Deleting outliers
train_df= train_df.drop(train_df[(train_df['GrLivArea']>4000) & (train_df['SalePrice']<300000)].index)

#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train_df['GrLivArea'], train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()


# # Assumptions of Linear Regression Model:
# 
# Linear regression is an analysis that assesses whether one or more predictor variables explain the dependent (criterion) variable.  
# The regression has five key assumptions:
# 
# * Linear relationship
# * Multivariate normality
# * No or little multicollinearity
# * No auto-correlation
# * Homoscedasticity

# # First things first: Lets Analyse Saleprice which is our dependent variable

# In[157]:


#Descriptive statistics summary
train_df['SalePrice'].describe()


# Let's create a histogram to see if the target variable is Normally distributed. If we want to create any linear model, it is essential that the features are normally distributed.

# In[158]:


#histogram
sns.distplot(train_df['SalePrice'] , fit=norm);


# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()

#skewness and kurtosis
print("Skewness: %f" % train_df['SalePrice'].skew())
print("Kurtosis: %f" % train_df['SalePrice'].kurt())


# # From the above graph, we can observe that the distribution:
# 
# * Deviate from the normal distribution
# * Have positive skewness
# * Show peakedness
# 
# The target variable is right skewed. As (linear) models love normally distributed data , we need to transform this variable and make it more normally distributed.
# 

# # Log-transformation of the target variable

# In[159]:



#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

#Check the new distribution 
sns.distplot(train_df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()


# The skew seems now corrected and the data appears more normally distributed.

# In[160]:


# most correlated features 
corrmat = train_df.corr()
top_corr_features = corrmat.index[abs(corrmat['SalePrice'])>0.5]
plt.figure(figsize=(10,10))
sns.heatmap(train_df[top_corr_features].corr(),annot = True);
top_corr_features


# We used heatmap here, so we can get the overview of all the features relationship:
# 
# In summary, we can conclude that:
# 
# 'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. Both relationships are positive, which means that as one variable increases, the other also increases. 
# 'OverallQual' and 'YearBuilt' also seem to be related with 'SalePrice'. The relationship seems to be stronger in the case of 'OverallQual', where the heat map shows how sales prices increase with the overall quality.
# We just analysed four variables, but there are many other that we should analyse. 
# 
# GarageCars and GarageArea are also some of the most strongly correlated variables.
# 
# Same goes for TotalBsmtSF and 1stFloor.
# 
# * Top correlated features are the ones which have more than 50% correlation with SalePrice

# # Scatter plot between 'SalePrice' and its correlated Variables

# In[161]:


sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_df[cols], size = 3)
plt.show();


# # Feature Engineering

# Let us concatenate the train and test data in the same dataframe

# In[162]:


ntrain = train_df.shape[0]
ntest = test_df.shape[0]
y_train = train_df.SalePrice.values
print("y_train shape is : {}".format(y_train.shape))
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


# # Calculate the percentage of missing values by each feature

# In[163]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)


# Visualize the missing values by histogram

# In[164]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# 
# According to the table, the below variables have more than 15% of the data missing:
# ('PoolQC','MiscFeature','Alley','Fence','FireplaceQu','LotFrontage')
# 
# 
# We can see that "GarageX" variables have he same number of missing data. 
# 
# The same logic applies to 'BsmtX' variables. The variables 'BsmtExposure', 'BsmtFinType1','BsmtQual','BsmtCond','BsmtFinType2' have similar percentages of missing data. 'MasVnrArea' and 'MasVnrType' have strong corelation with 'YearBuilt' and 'OverallQual'.
# 
# Finally 'Electrical' have only one null value.

# # Imputing the missing values

# We impute them by proceeding sequentially through features with missing values

# * PoolQC : data description says NA means "No Pool". That make sense, given the huge ratio of missing value (+99%) and majority of houses have no Pool at all in general.
# * MiscFeature : data description says NA means "no misc feature"
# * Alley : data description says NA means "no alley access"
# * Fence : data description says NA means "no fence"
# * FireplaceQu : data description says NA means "no fireplace"
# * LotFrontage : Since the area of each street connected to the house property most likely have a similar area to other houses in its neighborhood , we can fill in missing values by the median LotFrontage of the neighborhood.
# * GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
# * GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
# * BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having no basement
# * BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
# * MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
# * MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
# * Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
# * Functional : data description says NA means typical
# * Electrical : It has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
# * KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
# * Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
# * SaleType : Fill in again with most frequent which is "WD"
# * MSSubClass : Na most likely means No building class. We can replace missing values with None    

# In[165]:


all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


# Checking for remaining missing values

# In[166]:


#Check remaining missing values if any 
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# # Transforming some numerical variables that are really categorical

# In[167]:


#MSSubClass=The building class
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


# Label Encoding some categorical variables that may contain information in their ordering set

# In[168]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))


# Since area related features are very important to determine house prices, we add one more feature which is the total area of basement, first and second floor areas of each house

# In[169]:


# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# # Fixing Skewness

# In[170]:


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)


# # Box Cox Transformation of (highly) skewed features

# We use the scipy function boxcox1p which computes the Box-Cox transformation of  1+x .
# 
# Note that setting  λ=0  is equivalent to log1p used above for the target variable

# In[171]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])


# # Getting dummy categorical features

# In[172]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)


# Getting the new train and test sets.

# In[173]:


x_train = all_data[:ntrain]
x_test = all_data[ntrain:]
print("x_train shape is : {}".format(x_train.shape))
print("x_test shape is : {}".format(x_test.shape))
print("y_train shape is: {}".format(y_train.shape))


# # Splitting the data into training and test datasets

# In[174]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train,test_size = .3, random_state=0)
print("x_train shape is : {}".format(X_train.shape))
print("x_test shape is : {}".format(X_test.shape))
print("y_train shape is: {}".format(Y_train.shape))
print("y_train shape is: {}".format(Y_test.shape))


# # Modelling

# Importing the Required Libraries

# In[175]:


# importing all the required library for modeling here we are going to use statsmodels 
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge,  BayesianRidge, LassoLarsIC
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

import xgboost as xgb
import lightgbm as lgb


# # Regularization Models
# 
# What makes regression model more effective is its ability of regularizing. The term "regularizing" stands for models ability to structurally prevent overfitting by imposing a penalty on the coefficients.  We will also define a function that returns the cross-validation rmse error so we can evaluate our models and pick the best tuning parametr. The main tuning parameter for the regularization model is alpha - a regularization parameter that measures how flexible our model is. The higher the regularization the less prone our model will be to overfit. However it will also lose flexibility and might not capture all of the signal in the data.
# 
# We are going to look at the below two regularization techniques
# 
# * Ridge
# * Lasso

# Cross Validation Strategy
# 
# We use the cross_val_score function of Sklearn. However this function has not a shuffle attribute, we add then one line of code, in order to shuffle the dataset prior to cross-validation

# In[176]:


#Validation function
n_folds = 10
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, Y_train, scoring="neg_mean_squared_error", cv = 5))
    #scores = cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf)
    return(rmse)
# rmsle
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


# Set up Alpha values

# In[177]:


alphas_alt = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]


# # Ridge

# In[178]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ridge=Ridge()
parameters= {'alpha':[x for x in alphas_alt]}

ridge_reg=GridSearchCV(ridge, param_grid=parameters)
ridge_reg.fit(X_train,Y_train)
print("The best value of Alpha is: ",ridge_reg.best_params_,ridge_reg.best_score_)

cv_ridge_mean_list =[]
cv_ridge_std_list =[]
for alpha in alphas_alt:
    ridge_reg = rmsle_cv(Ridge(alpha = alpha))
    print("The alphas is : {}".format(alpha))
    print("Lasso Score mean is {:.4f}\n".format(ridge_reg.mean()))
    print("Lasso Score std is {:.4f}\n".format(ridge_reg.std()))
    cv_ridge_mean_list.append(ridge_reg.mean())
    cv_ridge_std_list.append(ridge_reg.std())


cv_ridge_mean = pd.Series(cv_ridge_mean_list, index = alphas_alt)
cv_ridge_std = pd.Series(cv_ridge_std_list, index = alphas_alt)
cv_ridge_mean.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmsle")
print("\nRdige score: {:.4f} ({:.4f})\n".format(cv_ridge_mean.min(), cv_ridge_std.min()))


# Note the U-ish shaped curve above. When alpha is too large the regularization is too strong and the model cannot capture all the complexities in the data. If however we let the model be too flexible (alpha small) the model begins to overfit. A value of alpha = 10 is about right based on the plot above.
# 
# So for the Ridge regression we get a rmsle of about 0.1174
# 
# Let' try out the Lasso model. We will do a slightly different approach here and use the built in Lasso CV to figure out the best alpha for us. For some reason the alphas in Lasso CV are really the inverse or the alphas in Ridge.
# 
# 

# # Lasso

# In[179]:


lasso=Lasso()
parameters= {'alpha':[x for x in alphas2]}

lasso_reg=GridSearchCV(lasso, param_grid=parameters)
lasso_reg.fit(X_train,Y_train)
print("The best value of Alpha is: ",lasso_reg.best_params_,lasso_reg.best_score_)

cv_lasso_mean_list =[]
cv_lasso_std_list =[]
for alpha in alphas2:
    lasso_reg = rmsle_cv(Lasso(alpha = alpha))
    print("The alphas is : {}".format(alpha))
    print("Lasso Score mean is {:.4f}\n".format(lasso_reg.mean()))
    print("Lasso Score std is {:.4f}\n".format(lasso_reg.std()))
    cv_lasso_mean_list.append(lasso_reg.mean())
    cv_lasso_std_list.append(lasso_reg.std())

cv_lasso_mean = pd.Series(cv_lasso_mean_list, index = alphas2)
cv_lasso_std = pd.Series(cv_lasso_std_list, index = alphas2)
cv_lasso_mean.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmsle")

print("\nLassoscore: {:.4f} ({:.4f})\n".format(cv_lasso_mean.min(), cv_ridge_std.min()))


# The lasso performs even better at aplha = 0.0005, so we'll just use this one to predict on the test set. Another neat thing about the Lasso is that it does feature selection for you - setting coefficients of features it deems unimportant to zero. Let's take a look at the coefficients:

# In[180]:


Lasso_model =Lasso(alpha=0.0005)
Lasso_model.fit(x_train,y_train)
y_pred_train=Lasso_model.predict(X_train)
y_pred_test=Lasso_model.predict(X_test)

print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(Y_train, y_pred_train))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(Y_test, y_pred_test)))) 


# In[181]:


coef = pd.Series(Lasso_model.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# Good job Lasso. One thing to note here however is that the features selected are not necessarily the "correct" ones - especially since there are a lot of collinear features in this dataset. One idea to try here is run Lasso a few times on boostrapped samples and see how stable the feature selection is.

# # Coeficients

# In[182]:


imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# The most important positive feature is GrLivArea - the above ground area by area square feet. This definitely sense. Then a few other location and quality features contributed positively. Some of the negative features make less sense and would be worth looking into more - it seems like they might come from unbalanced categorical variables.
# 
# Also note that unlike the feature importance you'd get from a random forest these are actual coefficients in your model - so you can say precisely why the predicted price is what it is. The only issue here is that we log_transformed both the target and the numeric features so the actual magnitudes are a bit hard to interpret.

# In[183]:


#let's look at the residuals as well:
plt.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":Lasso_model.predict(X_train), "true":Y_train})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")


# The residual plot looks pretty good.

# Let us use the lasso model for the predcitions and look at the RMSE for train and test data and submit the predcitions

# In[184]:



lasso_preds = np.expm1(Lasso_model.predict(X_test))
lasso_train = np.expm1(Lasso_model.predict(X_train))
print(rmsle(lasso_preds, Y_test))

print('Root Mean Square Error train = ' + str(np.sqrt(mean_squared_error(Y_train, lasso_train))))
print('Root Mean Square Error test = ' + str(np.sqrt(mean_squared_error(Y_test, lasso_preds)))) 


# # Final Prediction

# In[185]:


y_test=Lasso_model.predict(x_test)
predictions=np.expm1(y_test)


# In[186]:


submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
prediction = pd.DataFrame(predictions)
final_submission = pd.DataFrame({'Id':submission['Id'],'SalePrice':predictions})

final_submission.dropna(inplace=True)

final_submission['Id']=final_submission['Id'].astype(int)

final_submission.to_csv('submission1.csv', index=False)


# In[187]:


final_submission.head()

