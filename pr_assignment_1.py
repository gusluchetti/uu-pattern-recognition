
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import sklearn.metrics

#from sklearn.model_selection import cross_val_score
print('start')

mnist_data = pd.read_csv('mnist.csv').values

labels = mnist_data[:, 0]       # the true number written by the user
digits = mnist_data[:, 1:]      # the pixel value of the image 


# img_size = 28
# plt.imshow(digits[0].reshape(img_size, img_size))
# plt.show()


# create ink feature

ink = np.array([sum(row) for row in digits])

# compute mean for each digit class
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]

# compute standard deviation for each digit class
ink_std = [np.std(ink[labels == i]) for i in range(10)]

#print(np.size(ink), np.size(ink_mean), np.size(ink_std))
arr = np.array([ink_mean, ink_std])
#print(pd.DataFrame(arr))

# [TODO] make a table out of this array in our overleaf, explain the necessary things as written in the assignment
# stuff like 0 has the most ink and 1 the least etc, maybe point out how 2 has a lot more, but the stdev is pretty high compared to the values so we can't really say too much of note there
# perhaps apply some statistical test for significance? idk maybe that's overkill

scaled_ink = (ink - np.mean(ink))/np.std(ink)
transposed_scaled_ink = np.array([scaled_ink]).transpose()

def logreg_ink_only():
    

    # cv = 10 --> we use 10 folds
    # penalty = 'l1' --> lasso regularization
    # we need a solver that supports l1 penalty, saga also handles multinomial loss --> solver = 'saga'
    logreg_ink = sklearn.linear_model.LogisticRegressionCV(cv = 10, penalty='l1', solver='saga').fit(transposed_scaled_ink,labels)

    predictions_ink = logreg_ink.predict(transposed_scaled_ink)

    print(sklearn.metrics.confusion_matrix(labels,predictions_ink))
    print(sklearn.metrics.multilabel_confusion_matrix(labels,predictions_ink))
    print(sklearn.metrics.classification_report(labels,predictions_ink))

    # weird that there are labels with no correctly predicted samples (labels 4, 5, 6, 8) --> [TODO] ask professor about it


logreg_ink_only()






# other feature, go through each row of pixels, count how many times it reaches a non-zero pixel with a zero ink pixel separator --> if highest value of this is 2, feature is # of rows value 2 / # of rows value 1
# we do not care about rows with value 0 --> covered by ink feature
# problem: does not distinguish between 6 and 9 --> ink feature also does not distinguish between those in theory

#[TODO] implement this feature

#[TODO] do logregcv with both this and ink as features

# for part 2, when we need to consider each pixel as an individual feature
# [TODO] the entirety of part 2 lol

#df.info()

#print(df[400].sum())
#print(df.sum())

#print(df.loc[:, (df.sum() > 0).all()])

# [TODO] find a way to remove all pixels that always have constant value --> those columns will have stdev of 0, which will give a division by 0 when scaling