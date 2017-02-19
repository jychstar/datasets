Orginal dataset is held at http://ai.stanford.edu/~amaas/data/sentiment/. After unfolding, you get several files and two folders name "train" and "test". Within "train" folder, a "pos" folder( 12.5 k txt files with rating>= 7), a "neg" folder (12.5 k txt files with rating <= 3), a 'unsup' folder with intentionally omitted rating. 

In Udacity DLND course for sentiment analysis , all these labelled text files are already merge together for easy data input. 

- 25 k instances. Each instance is a review with multiple sentences
- 1 binary labels: "positive", "negative"

Obviously, the instances are not ready for use. We have to break each sentences into "bag of words", use their frequency to build a "succinct" dictionary.



workflow

1. use`collections.Counter().update(word)` to count the distinct words. get 74074 ones.
2. use `sorted` to get the most frequent 10 k words as a list called `vocab`
3. build a dictionary called `word2idx`, the content is the word's order. The reason of doing so is to accelerate retrieval of a word's index, because vocab.find(word) is just too slow and error-prone if not exist.
4. These 10 k words are used as the feature dimensions. Reformat the original reviews, so each review has 10 k "one hot encoded" features.

