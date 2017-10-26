Truth stroy: Titanic sank on 1912-4-15, killing 1502 out of 2224 passengers, a survival rate of 32.5%.

The dataset is held at https://www.kaggle.com/c/titanic

On 2017-3-16, I showed an Xap, a web-based platform of Exaptive, for the final interview. It still works and you can drag train.txt to the page: https://exaptive.city/xap/fe1dc1d0-0513-11e7-9a0d-372a3901145c

train.csv:

- 891 instances
- 12 attributes (including 1 binary lable)
- Age has 177 missing data
- survival rate 38.4%

test.csv:

- 418 instances

In **titanic_survival_exploration** of machine learning nanodegree, we have:

- predictions_0, always predicts not survive: 61.62%
- predictions_1, predicts all females survive: 78.68%
- predictions_2, add younger than 10 male survive: 79.35%
- predictions_3, add younger than 15, 2nd class above male survive: **80.02%**. I can reach 82.38% training accuracy with even more carefully handpick rules. 

To get data ready for machine learning algorithm, first we fill the missing age by the median value in each gender&class group. (alternatively, we can use other features to do a regression to predict the age)

However, machine learning algorithms don't achieve better results. One reason is the data size is still too small and randomness still matters a lot. With a single-hidden-layer neural network, the accuracy is only 62.7% for a train_test_split ratio of 0.3. 