This dataset is hosted at (https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) It is hourly and daily records of the riders in Capital Bikeshare System at Washington,DC. The bike sharing program begins at 2010, owned by local government and has over 2 million riderships annually. Invest \$5M , operate cost 2M for 100 stations. Earned revenue covers 50% operating cost, which means it relies on public money to benefit public environment. 

- 17379 instances, ~ 24\*365\*2. (2011-2012)
- 16 attributes

How do I know this dataset?  The project 1 of deep learning foundation nanodegree uses this dataset for practicing home-made neural network. 

## Data preprocessing

- data observation

- `pd.get_dummies()` for 5 categorical variables: season (4), weathersit (clear, mist, light rain, heavy rain), mnth(12), hr(24), weekday(7)

- `pd.drop(), pd.concat()` for above 5 variables and 4 other unimportant factors: atemp, datedly, instant, working day.

- For non-catergorical data, scale by mean and std

- split data by saving last 21 days for testing, another 60 days for validation

- pick ['cnt', 'casual', 'registered'] as target, the remaining 56 columns are features

```python
# Save the last 21 days 
test_data = data[-21*24:]
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
```

## Data mining

1. define a class called "NeuralNetwork" with three method:\__init\__, train, run. This is the backbone of this project
2. train and run the network with timely feedback by `sys.stdout.write()`
3. visualize the result
4. unittest to test the codes. 5 tests are customizedly defined in a class called "TestMethods" 