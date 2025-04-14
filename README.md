# California House Price Prediciton using Regression Models

## 1. Introduction

The focus of this project is to build several regression models to accurately predict the house median value in california. The regression models used are Linear Regression, Ridge Regression and Decision Tree. Since, there is a possibility that the Machine learning models suffer from overfitting, hyperparameter tuning is implemented to all three models. The results are evaluated using evaluation metrics like Mean Squared Error and $R^2$ score.

## Libraries and Dependencies

The libraries that are necessary for this project are listed below:
[![Screenshot-2025-04-13-105113.png](https://i.postimg.cc/76S8VnyL/Screenshot-2025-04-13-105113.png)](https://postimg.cc/DJfMfs0V)

## 2. Preprocessing

The dataset that is used for this project is California House prices dataset from Kaggle. The dataset contains information from 1990 California census. This dataset contains 20640 rows and 10 columns. These 10 columns are divided into 9 features and 1 target variable called the "median house value" as shown in the figure below:

[![Screenshot-2025-04-13-111348.png](https://i.postimg.cc/7LGdXny2/Screenshot-2025-04-13-111348.png)](https://postimg.cc/Tyxt3m1d)

### Exploratory Data Analysis (EDA)

Exploratory Data Analysis is important to better understand the dataset and relation between features and target variables. The figure below shows the individual features and their data type. The table also shows the number of non-null values/observations for each feature. Upon careful analysis, there are two features that caught the attention:

1. The feature called 'total_bedrooms' contain some missing values because the total number of observation is 20433.
2. The data type of the feature, 'ocean_proximity' is an object which specifies that it is not a numerical feature.

### Numerical Features
As the preprocessing techniques for numerical features and categorical features is different, The feature columns are divided into 8 numerical columns and 1 categorical column. The table below gives statistical information about the individual features excluding 'ocean_proximity' because it is a categorical feature. Additionally, The table also contains the statistical information of the target column, 'median_house_value' to check if the target variable has any missing values.

![image](https://github.com/user-attachments/assets/58a7d520-779f-4da2-9989-2df280256146)

The table shows that 'total_bedrooms' contain some missing values. The feature, 'total_rooms' consist of a maximum value of 39320 but however its third quartile value is 3148. This means the feature consists of some outliers as the maximum value is exponentially high. In order to visualize this information, a histogram is plotted.

![Screenshot 2025-04-13 123953](https://github.com/user-attachments/assets/8fa69b86-83b1-41cb-bb49-bd6535fc0ec3)

The histogram shows that the features 'median_house_value' and 'housing_median_age' consists of some outliers as there is a negative skewness in both the features which means that some values are exponentially higher than the others. These are called outliers. 


#### Missing values
As stated previously, the dataset has 9 features and 1 target. For preprocessing, the dataset is split into train and test set. The recommended split is 80% of the dataset as train set and 20% as test set. 
In order to deal with the missing values, a simple imputer is used which takes the median value of itself and replaces the nan values with the median value. In the case of 'total_bedrooms', the missing value will have a value of 435.00 as that is the median. 

#### Feature Scaling
After dealing with the missing values in the numerical columns, it is important to do feature scaling to the numerical data. Feature scaling is important because it evens out the magnitudes of different features. For example, features like 'median_income’ has fractional value of 0.53 while 'total_rooms’ contains values that are very large like more than 20,000. There are two common feature scaling methods, Standardisation and Normalisation(MinMaxScaling). It is evident that feature like ’total rooms’ consist of some outliers. Due to its sensitivity to outliers, Normalisation is not recommended for feature scaling in this dataset. 
Standardisation is considered for feature scaling.

### Categorical Feature

The dataset contains one categorical feature called 'ocean_proximity'. The figure below shows that this feature doesn't have missing values. So, the use of Imputer is not necessary and encoding can be used straightaway. 

![Screenshot 2025-04-13 123011](https://github.com/user-attachments/assets/10f4b93e-b212-466c-90dc-0ed3d9fe9b2e)

#### Encoding
For the Machine learning models to use the categorical features, it needs to be converted into numerical values by using a
encoder. There are two common encoding methods: Label encoding and one-hot encoding. The categorical feature, ’ocean
proximity’ is nominal which means it does not have any order. One-hot encoding is used when the order does not matter.
In this case, one-hot encoding is suitable for handling the categorical variable.
This concludes the pre-processing section. Finally, the numerical and categorical columns are combined into a single
feature matrix again.


## 3. Methodology
This project focuses on the prediction of the median house value with three different regression models,
that is Polynomial Regression Model, Ridge Regression Model and Decision Tree Model. Polynomial Regression Model outperforms
both Ridge regression model and Decision Tree Model with no signs of overfitting or underfitting.
Ridge regression is a linear regression with the $l^2$ regularization term. Cost function of ridge regression is slightly different
than a linear regression model with the addition of a regularisation function. In mathematical terms the Polynomial regression model is defined as:

For single variable,
<div align="center">

$$
y = \sum_{j=0}^{d} \theta_j x^j + \varepsilon
$$
</div>

where,
      $y$ : predicted target value </br>
      $x$ : single feature        
      $d$ : degree of the polynomial</br>
      $\theta_j$ : parameter for the $j^{ith}$ power of x</br>

Alternatively, it can be written as:

<div align="center">
  
$$
y = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3 + \cdots + \theta_d x^d + \varepsilon
$$

</div>
where,</br>

$\theta = (X^T X)^{-1} X^T y$

The cost function of Polynomial Regression is written as:
<div align="center">
  
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
$$
</div>

To judge how well our models are performing, Two evaluation metrics are used:

1.Mean Squared Error

$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

The goal is to minimise this value. MSE must be 0  (for a perfect model) or positive, due to any errors being squared.

2. $R^2$  Score

This Explains how closely our predictions match the real data.
Generally in the range  $0≤R^2≤1$, where  1  is a perfect fit.

## 4.Experiments
### 4.1 Hyperparameter Tuning
When training the models, Instead of changing the strength of hyper-parameter of each model manually, a framework is created using loop method to automatically increment the hyper-parameter with each iteration to select the best hyper-parameters of each model. Then, the best hyper-parameters for the model is used to make predictions and get the Mean squared Error and $R^2$ score.

#### 4.1.1 Polynomial Regression
 For Linear Regression, the degree of polynomial is a key hyper-parameter. To find the best degree, a loop iterates five times, increasing the polynomial degree with each iteration. Standardization occurs after polynomial conversion because degree of polynomial aims to amplify the signal, which is less effective if scaling reduces feature values beforehand.
 The graph below represents Mean Squared Error of train and test data are the closest when the degree of polynomial is 2.
![Screenshot 2025-04-13 155725](https://github.com/user-attachments/assets/9776f481-3a91-48f7-bfed-dce1c8cbb0c9)

A visual representation of how well our predictions are fitting the data are shown in the figure below. Due to having high number of dimensions, the graph is plotted selecting the feature that explains the most variation in the house prices, that is 'median_income'. The x-axis represents 'median_income' and Y-axis represents the target feature, 'median_house_value' :
![Screenshot 2025-04-14 194734](https://github.com/user-attachments/assets/8e5eff68-38c5-4e21-bfb8-86dd65762c29)


#### 4.1.2 Decision Tree 
Decision Trees is a non parametric model which consist of several hyper-parameters like 'max depth', 'min samples split' and 'min samples leaf'. These hyper-parameters are tuned using a cross validation method. This method splits the train and test data into k folds to get the best split that minimizes Mean Squared Error.

![Screenshot 2025-04-13 160051](https://github.com/user-attachments/assets/7c2450e9-23d4-417b-9e91-f89643608a36)

#### 4.1.3 Ridge Regression
Ridge regression model being similar to Linear Regression, the second degree of polynomial features is applied as it yielded the best results for linear regression. Also, ridge regression model consists of regularization weight (${\alpha}$) which is also a hyper-parameter. The loop method is applied to this model that  increases $alpha$ value with each iteration. The model is trained on train set and predictions are made on train set with the value of ${\alpha}$  changing through every iteration. There is no sign of overfitting and underfitting when the $alpha$ is 3 as shown in the figure below.

![Screenshot 2025-04-14 211357](https://github.com/user-attachments/assets/7bc7d161-6350-422a-9fd3-53ac70f07437)

![Screenshot 2025-04-14 211409](https://github.com/user-attachments/assets/6e32c172-c0e7-4b4c-af2c-5ff6f92cb771)


