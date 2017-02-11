The [Eigenface approach](https://en.wikipedia.org/wiki/Eigenface) was developed by Sirovich and Kirby (1987), who showed that principal component analysis could be used. 

"Labeled faces in the wild" dataset is original hosted at http://vis-www.cs.umass.edu/lfw/. It contains more than 13 k images of faces.

The dataset used by `sklearn.datasets.fetch_lfw_people`is a "funneled" version, created on 2008-2-4, 220 MB:

- 1288 instances
- 1850 features or 50\*37 features
- target has 7 classes: ['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush' 'Gerhard Schroeder' 'Hugo Chavez' 'Tony Blair']

## workflow in `face_recognition.ipynb`

1. Automatically download LFW metadata to `home/scikit_learn_data` folder, which takes about 25 minutes for my computer.
2. `train_test_split(X,y,test_size=0.25)`
3. `pca = PCA(n_components= 150).fit(X_train)`
4. `eigenfaces = pca.components_.reshape((150, 50, 37))` 
5. `np.cumsum(pca.explained_variance_ratio_[0:12])`, it shows that  only the top 4 eigenfaces already present 50% of the variances. After that, the cumulation increases very slowly
6. `pca.transform`
7. use(X_train_pca,y_train) to do GridSearchCV
8. `sklearn.metrics.classification_report` to see a detailed prediction result. We can see Bush has the highest f1 score (0.91) while the average is 0.86. Alternatively, `confusion_matrix` is very handy.
9. imshow testing pictures and eigen faces