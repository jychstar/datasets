This dataset is hosted at https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset. It is hourly and daily records of the riders in Capital Bikeshare System at Washington,DC. The bike sharing program begins at 2010, owned by local government and has over 2 million riderships annually. Invest \$5M , operate cost 2M for 100 stations. Earned revenue covers 50% operating cost, which means it relies on public money to benefit public environment. 

- 17379 instances, ~ 24\*365\*2. (2011-2012)
- 16 attributes

## workflow

- data observation

- pd.get_dummies() for 5 categorical variables

- pd.drop for 5+4 variables, 

- data scaling by mean and std

- split data by saving last 21 days for testing, another 60 days for validation

- pick ['cnt', 'casual', 'registered'] as target, the remaining 56 columns are features
