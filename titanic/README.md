## About the Xap

Backstory: An OKC-based start-up company Exaptive is building a web-based platform for data scientist. The key idea is to split the data pipeline into reusable modules, especially for the data visualization on the web. At the same time, you can still write your Python or R codes to do the data mining. 

From 2017-2-15 to 2017-3-16, I used their platform to build an **Xap:** https://exaptive.city/xap/fe1dc1d0-0513-11e7-9a0d-372a3901145c

It still works. You can drag the [train.txt](https://s3-us-west-2.amazonaws.com/jychstar/train.csv) into the webpage and see the magic!

You will see 2 charts. Both of them take advantage of JavaScript library. The bar plot uses JS library dimple. The pie plot uses JS library plotly.  

The block diagram looks like this:

![](https://lh3.googleusercontent.com/UReI5mDaSYLMMNakhu26DX8L9j9rJLIpiehRz-320rQ-a23g2xEDQQTCIGhahTHW1M5sNNh85ygwJMiSKo02-sUI9TR2ffCTTu_JxoyzGZhOW6Jcre31Vb1PZ7VKz-MibYIjSH0PjelMOwON_E9Bk0PHr0KVha-dZk65jHC07ZM3ZRt8i7YMyc80acntNq-wVJjeBezGu0N36qb7ly5mpeM3RwpxpbpWkXcNlIJgvOLYtXsPkfPjNd9KGPEW1REZk94TdSgUoEYSPthlO0ps2oVhWYUmrJDmWsV4LTu8brlJVvNn7Wq6ghpGoBz1Em08dN7Ngn61nZnuhpOOgQQMAtxAR6zYjPCvjXtkweTCHa47yvz3z6oSNMPioLlJbwhiqXdUHTX7odLEubAePwgYhyktBlFPb1o-wwqjwDo8Ob8Hqc-GC678JtH7ZW2nHdybCCz6tkLpkHLAwCj41HvQSoFhEMF-InWEN0y0ThrMGGAtNACmBagkPtc2i-4qhp07W00-fxxMnwlYRSJSXW22HjXXaWgt1PWUhHdSH7Qh2h05SXyQT-as4mdJ318jTcqFIycLRGM5jpDzDTDuQG-vxrTEzSJQz7ITd5_jACx93A=w2390-h1494-no)

The dataflow is very intuitive. However, because every component (block) is built from scratch by the few developers, it is really **difficult to scale** and compete with other similar software such as **Alteryx**.  There is always **a tradeoff between customization and scaling**. 

---

## About the dataset

History: Titanic sank on 1912-4-15, killing 1502 out of 2224 passengers, a survival rate of 32.5%.

Original dataset is held at https://www.kaggle.com/c/titanic

train.csv:

- 891 instances
- 12 attributes (including 1 binary lable)
- Age has 177 missing data
- survival rate 38.4%

test.csv:

- 418 instances

## About machine learning

In **titanic_survival_exploration** of machine learning nanodegree, we have:

- predictions_0, always predicts not survive: 61.62%
- predictions_1, predicts all females survive: 78.68%
- predictions_2, add younger than 10 male survive: 79.35%
- predictions_3, add younger than 15, 2nd class above male survive: **80.02%**. I can reach 82.38% training accuracy with even more carefully handpick rules. 

To get data ready for machine learning algorithm, first we fill the missing age by the median value in each gender&class group. (alternatively, we can use other features to do a regression to predict the age)

However, machine learning algorithms don't achieve better results. One reason is **the data size is still too small** and randomness still matters a lot. With a single-hidden-layer neural network, the accuracy is only 62.7% for a train_test_split ratio of 0.3. 