Enron Email Dataset is original hosted at https://www.cs.cmu.edu/~./enron/. The earliest version is 2014-3-2, when Enron was finalizing its bankcrupt. The newest version is  enron_mail_20150507.tgz, 423 MB.

After unzipping this tgz file, you get a folder "maildir" which is 1.42 GB and has 150 folders. Each foler is name by a person, then you have detail email text in various categories. Reading this email is extremely boring! **No one is interested in digging into the scandal unless you get paid to do this professional job.** 

I am trying to review what I have learned during "Intro to Machine Learning" course. This is a good chance for me to fully apprecaite this dataset. Udacity's codes are hosted at https://github.com/udacity/ud120-projects.git

| code file                     | accuracy, time | note                                     |
| ----------------------------- | -------------- | ---------------------------------------- |
| naive_bayes/nb_author_id.py   | 0.973, 1s      |                                          |
| svm/svm_author_id.py          | 0.991, 120s    | kernel="rbf", C=10000                    |
| decision_tree/dt_author_id.py | 0.977,  63s    | min_samples_split=40                     |
| decision_tree/dt_author_id.py | 0.967, 4s      | SelectPercentile(..., percentile=1), default is 10 |

Last one is to reduce features from 10% to 1%.  The actual dirty job is done by tools/email_preprocess.py, which use `sklearn.feature_extraction.text.TfidfVectorizer` to vectorize text and use `sklearn.feature_selection.SelectPercentile` for feature reduction. What'more, the bigger picture is: the 17.6 k training instances and binary labels are already well packed there for you to use. 

## Datasets and Questions

```python
with open("final_project/final_project_dataset.pkl", "r") as f:
  enron_data = pickle.load(f)  # dict, 146 keys
```

So enro_data is a diectionary with 146 keys. Each key is a person's name (capitalized).The corresponding content is another 21-key dictionary, in which you can see poi, salary, bonus, total_payments, etc.

