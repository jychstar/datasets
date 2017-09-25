kaggle project: https://www.kaggle.com/c/DontGetKicked

This project is trying to predict if a car purchased at auction is a lemon (IsBadBuy).  Among the 72983 training samples, only 12.3% are positive results. This makes the dataset highly skewed. In other words, even we blindly predit all outputs are negative, we still get a baseline accuracy of **87.7%**. So f-score should be used as evaluation metric to represent more interesting precision and recall value.

## 1. data exploration

Among 32 input features, 13 have no missing value, 11 have slightly missing value. These features are not completely independent. For example, VehYear and VehicleAge indicate the exact same thing. The 8 Auction/Retail prices are highly correlated to each other as well as VehBCost and WarantyCost. Model/subModel are related to make/Size/natioality/TopThreeAmericanName.

To explore the relation between a single input feature and the target feature(IsBadBuy), I use `seaborn.boxplot` for numerical features and `seaborn.factorplot` for categorical features. With intution and understanding,  influential features are picked as follows:

- Auction: ADESA gives more BadBuy
- VehYear, VehicleAge: both represent the same thing, BadBuy has larger vehicle age
- Make
- Color: **missing 8 data**   
- Transmission: data dominant by automation, **missing 1 data**   
- WheelTypeID: type 3 leads to more BadBuy; **missing 3000 +data**
- VehOdo: BadBuy has larger odometer
- Naitonality: data dominate by American (61k),**missing 5 data**    
- Size: **missing 5 data**
- TopThreeAmericanName: **missing 5 data**
- Price: BadBuy has lower price in whatever type, **missing 13 data**
- VehBCost: BadBuy has slightly lower buy cost, not significant
- WarrentyCost: BadBuy has higher WarrentyCost
- IsOnlineSale: online sale has lower badBuy, but offline is dominate. (71k)

I also plot a correlation matrix for the numerical features and find VehBCost is highly related to the Auction price. One feature could be drop to faciliatte the training.

## 2. data preprocessing

I divided features into 2 groups: categorical and numberical. 

To build the training set:

1. try different combination of features, fill na or drop na
2. scale the numerical data
3. one-hot encode the categorical data
4. drop and original categorical features, separate the labelled feature.

## 3. data mining

Use train/test = 7:3 to split the data for model validation. 

### 3.1 first try

Initially, I use 10 features without "WheelTypeID" due to more than 3000 missing data. 

```python
feature_cate =['Auction', 'Make', 'Color', 'Transmission', 'Nationality','Size', 'TopThreeAmericanName','IsOnlineSale']
feature_num = ['VehicleAge','VehBCost']
```

I try 3 different machine learning algorithms from `sklearn`, the scores are:

| ML algorithm  | accuracy | f_0.5 | f_2   |
| ------------- | -------- | ----- | ----- |
| decision tree | 0.790    | 0.166 | 0.175 |
| random forest | 0.856    | 0.110 | 0.056 |
| SVM           | 0.880    | 0.0   | 0.0   |

SVM beats the baseline of 0.877, but the F score is zero! This means **no true positive**.

The equation of f score is: $F_\beta =\frac{ (1+\beta^2)  \times precision * recall)} { \beta^2precision + recall}$

And: ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/26106935459abe7c266f7b1ebfa2a824b334c807), ![](https://wikimedia.org/api/rest_v1/media/math/render/svg/4c233366865312bc99c832d1475e152c5074891b)

F0.5 focus more on precision, and F2 weights more on recall.

### 3. 2 review data and retry

After a lot of attempts, including buidling models based on individual features, I found most feature stuck at 86~87% accuracy without true positive.  The only exception is **WheelTypeID**, which can achieve 90.1% accuracy. To deal with its missing data, first I tried to correlate it to the "Make" or "Model" but didn't found useful correlation. Then I tried to assign the NA to 0.0 and it worked.

I tried different combination of features, the best I got is:

```python
feature_cate = ['Auction','Size',"WheelTypeID",'TopThreeAmericanName']
feature_num = ['VehicleAge','VehBCost']
```

the corresponding score is:

|           | accuracy | f_0.5 | f_2   |
| --------- | -------- | ----- | ----- |
| SVM (rbf) | 0.902    | 0.551 | 0.270 |

## 4 future improvement

There are several ways to further improve the model:

1. currently, I simply drop the few na in several features. It is better to fill na with proper value if some correlation can be found. 
2. try different combinations of features.
3. get higher quality data features such as carfax report. Although it requries additional cost, but the cost is relatively small compared to the transaction price.  
4. retune the model parameters, or change to other models such as Neural Network or AdaBoost. 