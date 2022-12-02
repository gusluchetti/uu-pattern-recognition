
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import timeit

#from sklearn.model_selection import cross_val_score
print('start')


mnist_data = pd.read_csv('mnist.csv').values

labels = mnist_data[:, 0]       # the true number written by the user
digits = mnist_data[:, 1:]      # the pixel value of the image 


# img_size = 28
# plt.imshow(digits[1].reshape(img_size, img_size))
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
    
# WE DO NOT ACTUALLY USE CROSS VALIDATION IN PART 1 OF THE ASSIGNMENT
    # cv = 10 --> we use 10 folds
    # penalty = 'l1' --> lasso regularization
    # we need a solver that supports l1 penalty, saga also handles multinomial loss --> solver = 'saga'
 #   logreg_ink = sklearn.linear_model.LogisticRegressionCV(cv = 10, penalty='l1', solver='saga').fit(transposed_scaled_ink,labels)


 # we use logistic regression with default values for hyperparameter C
    logreg_ink = sklearn.linear_model.LogisticRegression(penalty='l1', solver='saga').fit(transposed_scaled_ink,labels)
    predictions_ink = logreg_ink.predict(transposed_scaled_ink)

    print("ink only confusion matrix")
    print(sklearn.metrics.confusion_matrix(labels,predictions_ink))
    print("ink only multilabel confusion matrix")
    print(sklearn.metrics.multilabel_confusion_matrix(labels,predictions_ink))
    print("ink only classification report")
    print(sklearn.metrics.classification_report(labels,predictions_ink))

    # weird that there are labels with no correctly predicted samples (labels 4, 5, 6, 8) --> [TODO] ask professor about it


#logreg_ink_only()

def h_pass(samples = len(digits), example = None):
    
    minsequence = 1     # how many pixels have to be colored or empty in a row for it to be considered a change of color
    max_count = 6       # 6 is the smallest possible as determined by trial and error

    horizontal_passthrough = np.zeros(samples*max_count)
    horizontal_passthrough = np.reshape(horizontal_passthrough, (samples,max_count))    # making a 2D array of 'samples' rows and 'max_count' columns

    print("starting hpass")

    for i in range(samples):
        h_passthrough_sample = np.zeros(max_count)
        for row in range(28):  
            count = 0
            painted_section = False
            for pixel_index in range(28):
                pixel = digits[i][28*row + pixel_index]
                if painted_section == False:
                    if pixel != 0:
                        painted_section = True
                        count += 1
                    #    print("started counting on pixel", 28*row + pixel_index, "with value", pixel)
                else:
                    if pixel == 0:
                        painted_section = False
                        # print("went back on pixel", 28*row + pixel_index, "with value", pixel)
            if count >= 0:
                h_passthrough_sample[count] += 1
                if example != None:
                    if i == example:
                        print("sample",i,"row", row, "at count",count)
        horizontal_passthrough[i] = h_passthrough_sample

    # THIS IS OLD CODE TO ADD A MINSEQUENCE PARAMETER TO THE HPASS FEATURE

    # for i in range(samples):
    #     h_passthrough_sample = np.zeros(max_count)
    #     for row in range(28):
    #         count = 0
    #         sequence_count = 1
    #         painted_section = False
    #         for pixel_index in range(28):
    #             pixel = digits[i][28*row + pixel_index]
    #             if painted_section == False:
    #                 if pixel != 0:
    #                     if sequence_count >= minsequence:
    #                         painted_section = True
    #                         count += 1
    #                         sequence_count = 1
    #                     else:
    #                         sequence_count += 1
    #             else:
    #                 if pixel == 0:
    #                     if sequence_count >= minsequence:
    #                         painted_section = False
    #                         sequence_count = 1
    #                     else:
    #                         sequence_count += 1
    #         # if count > 0:
    #         #     h_passthrough_sample[count-1] += 1
    #         if count >= 0:
    #             h_passthrough_sample[count] += 1
    #     horizontal_passthrough[i] = h_passthrough_sample
    
    if example != None:
        print (horizontal_passthrough[example])

    return horizontal_passthrough

def h_pass_ratio(numbers = (2,1), samples = len(digits), example = None):

    # samples = len(digits)
    
    minsequence = 1     # how many pixels have to be colored or empty in a row for it to be considered a change of color
    max_count = 6       # 6 is the smallest possible as determined by trial and error

    horizontal_passthrough = np.zeros(samples)
    horizontal_passthrough = np.reshape(horizontal_passthrough, (samples,max_count))    # making a 2D array of 'samples' rows and 'max_count' columns

   # print(len(digits[0]))

    print("starting hpass")

    for i in range(samples):
        h_passthrough_sample = np.zeros(max_count)
        for row in range(28):  
            count = 0
            painted_section = False
            for pixel_index in range(28):
                pixel = digits[i][28*row + pixel_index]
                if painted_section == False:
                    if pixel != 0:
                        painted_section = True
                        count += 1
                    #    print("started counting on pixel", 28*row + pixel_index, "with value", pixel)
                else:
                    if pixel == 0:
                        painted_section = False
                        # print("went back on pixel", 28*row + pixel_index, "with value", pixel)
            if count >= 0:
                h_passthrough_sample[count] += 1
                if example != None:
                    if i == example:
                        print("sample",i,"row", row, "at count",count)
        ratio = h_passthrough_sample[numbers[0]] / h_passthrough_sample[numbers[1]]      # by default this gives ratio 2 to 1
        horizontal_passthrough[i] = ratio

    
    if example != None:
        print (horizontal_passthrough[example])

    return horizontal_passthrough


def make_img(index):
    img_size = 28
    plt.imshow(digits[index].reshape(img_size, img_size))
    plt.show()




# provide a specific example
#example_img = 7
#h_pass(samples = example_img+1, example = example_img)
#make_img(example_img)

hpass_matrix = h_pass(samples = 42000)
print(hpass_matrix)

#hpass_ratios = h_pass_ratio(samples = 10)

# justification for excluding zeroes: they only show how large or small the handwriting is, 
# we want the ratio within the number itself, and there is no number traditionally written with a complete horizontal gap

#print(hpass_ratios)

# [TODO] make a pandas dataframe of hpass, make a bar chart with 0 to 6 on x axis, and number of samples that had it as highest as y-axis
# [TODO] have a bar chart with 0 to 6 on x axis, and how many total rows with that number across all samples on y-axis

#print(len(hpass))



#hpass_transposed = hpass.transpose()

# [TODO] need to scale and transform the hpass matrix so it can fit into the logreg model
def logreg_hpass_only():
    
 # WE DO NOT USE CROSS-VALIDATION ON PART 1
    # cv = 10 --> we use 10 folds
    # penalty = 'l1' --> lasso regularization
    # we need a solver that supports l1 penalty, saga also handles multinomial loss --> solver = 'saga'
 #   logreg_hpass = sklearn.linear_model.LogisticRegressionCV(cv = 2, penalty='l1', solver='saga').fit(hpass_matrix,labels)
    logreg_hpass = sklearn.linear_model.LogisticRegression(penalty='l1', solver='saga').fit(hpass_matrix,labels)

    predictions_hpass = logreg_hpass.predict(hpass_matrix)

    print("hpass confusion matrix")
    print(sklearn.metrics.confusion_matrix(labels,predictions_hpass))
    print("hpass multilabel confusion matrix")
    print(sklearn.metrics.multilabel_confusion_matrix(labels,predictions_hpass))
    print("hpass classification report")
    print(sklearn.metrics.classification_report(labels,predictions_hpass))


tic=timeit.default_timer()


print("INK ONLY RESULTS:")
logreg_ink_only()

toc=timeit.default_timer()

print("ink only completed in time", toc-tic)

tic=timeit.default_timer()

print("HPASS ONLY RESULTS:")
logreg_hpass_only()

toc=timeit.default_timer()

print("hpass only completed in time", toc-tic)

# make a count of how many rows per number: how many rows with 1, with 2, with 3, with 4, with 5 --> see which is the highest number

# other feature, go through each row of pixels, count how many times it reaches a non-zero pixel with a zero ink pixel separator --> if highest value of this is 2, feature is # of rows value 2 / # of rows value 1
# we do not care about rows with value 0 --> covered by ink feature
# problem: does not distinguish between 6 and 9 --> ink feature also does not distinguish between those in theory


# for part 2, when we need to consider each pixel as an individual feature
# [TODO] the entirety of part 2 lol

#df.info()

#print(df[400].sum())
#print(df.sum())

#print(df.loc[:, (df.sum() > 0).all()])

# [TODO] find a way to remove all pixels that always have constant value --> those columns will have stdev of 0, which will give a division by 0 when scaling