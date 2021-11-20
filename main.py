
import pandas as pd
import numpy as np
import os
import time
from sklearn import metrics  # only for test the algorithm performance evaluation


# read the csv file
def readCSV(filename):
    result_csv = pd.read_csv(filename)
    result_data = pd.DataFrame(result_csv)
    return result_data


# Convert text data into numbers for easy calculation of Euclidean distance
def data_transform(data):
    # Convert these column data to numbers
    for j in range(data.shape[1]):
        for i in range(data.shape[0]):
            if data.iloc[i, j] == '5more' or data.iloc[i, j] == 'more':
                data.iloc[i, j] = 5
            if data.iloc[i, j] == 'vhigh' or data.iloc[i, j] == 'big' or data.iloc[i, j] == '4':
                data.iloc[i, j] = 4
            elif data.iloc[i, j] == 'high' or data.iloc[i, j] == 'big' or data.iloc[i, j] == '3':
                data.iloc[i, j] = 3
            elif data.iloc[i, j] == 'med' or data.iloc[i, j] == '2':
                data.iloc[i, j] = 2
            elif data.iloc[i, j] == 'low' or data.iloc[i, j] == 'small':
                data.iloc[i, j] = 1
    return data


def data_decomposition(data_file):
    data = readCSV(data_file)
    transformed_data = data_transform(data)
    train = transformed_data[:int(len(transformed_data) * 0.8)]
    test = transformed_data[int(len(transformed_data) * 0.8):]
    train_y = train.evaluation
    train_x = train.drop('evaluation', axis=1)
    # unacc and ACC are digitized as -1 and 1 respectively
    def fun(x):
        if x == 'unacc':
            return -1
        else:
            return 1

    train_y = train_y.apply(lambda x: fun(x))
    test_y = test.evaluation
    test_y = test_y.apply(lambda x: fun(x))
    test_x = test.drop('evaluation', axis=1)
    return train_x, train_y, test_x, test_y


# use k-Nearest Neighbor algorithm
# train_x represents the training set we use, excluding the column 'evaluation'
# train_y represents the column of the training set
# test_x repersents the test set we use
# we test k = 1 to 20 and we find when k=6 is selected, the highest accuracy is achieved
def Knn(train_x, train_y, test_x):
    # First, deal with the data
    train_group = train_x.values
    labels = train_y.tolist()
    test_x = test_x.values
    predict = []
    newInput = test_x[0]
    for i in range(test_x.shape[0]):
        newInput = test_x[i]
        # Step 1: the following copy numSamples rows for train_group
        diff = np.tile(newInput, (train_group.shape[0], 1)) - train_group  # diff by element
        squareDiff = diff ** 2
        squareSum = squareDiff.sum(axis=1)  # add up by row
        distance = squareSum ** 0.5  # Take the square root and get the Euclidean distance
        # Step 2: Sort these distances
        sortedDistance = distance.argsort()
        classCount = {}
        for i in range(6):
            # Step 3: choose k nearest neighbors
            voteLabel = labels[sortedDistance[i]]
            # Step 4: Count the number of occurrences of classes in k nearest neighbors
            # when the key voteLabel is not in dictionary classCount, get()
            # will return 0
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        # Step 5: return the most frequently occurring category label
        maxCount = 0
        maxIndex = 0
        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                maxIndex = key
        predict.append(maxIndex)
    return predict


# use Naive_Bayes algorithm
# train_x represents the training dataset we use
# train_y represents the column 'evaluation' of training dataset
# test_x represents the test dataset we use
def Naive_Bayes(train_x, train_y, test_x):
    # first compute the prior probability for each class
    P_acc = train_y.tolist().count(1) / train_y.shape[0]
    P_unacc = train_y.tolist().count(-1) / train_y.shape[0]
    # We create two matrices to store 𝑃 (𝐱|𝐶𝑖) for 𝑖 = 1, 2
    # Doing so can greatly reduce the time complexity of the algorithm
    P1 = np.zeros((6, 6))
    P2 = np.zeros((6, 6))
    # store the output
    predict = []
    # compute every 𝑃 (𝐱|𝐶𝑖) for 𝑖 = 1, 2
    for attribute in range(6):
        for index in range(train_x.shape[0]):
            ans = train_x.iloc[index, attribute]
            if train_y[index] == 1:
                P1[attribute][ans] = P1[attribute][ans] + 1
            else:
                P2[attribute][ans] = P2[attribute][ans] + 1
    # apply the Laplace correction
    P1 = (P1 + 1) / (train_y.tolist().count(1) + 2)
    P2 = (P2 + 1) / (train_y.tolist().count(-1) + 2)
    # process the test set and print the predicted values
    for i in range(test_x.shape[0]):
        Px_1 = P_acc
        Px_2 = P_unacc
        newInput = test_x.iloc[i, :]
        for i in range(6):
            value = newInput[i]
            Px_1 = Px_1 * P1[i][value]
            Px_2 = Px_2 * P2[i][value]
        if Px_1 > Px_2:
            predict.append(1)
        else:
            predict.append(-1)
    return predict


# use perceptron algorithm
# train_x represents the training set we use, excluding the column 'evaluation'
# train_y represents the column of the training set
# test_x repersents the test set we use
# we set the learning rate alpha = 0.1
def Perceptron(train_x, train_y, test_x):
    # First, deal with the data
    train_x.insert(0, 'I0', 1)
    train_x = np.matrix(train_x)
    test_x.insert(0, 'I0', 1)
    test_x = np.matrix(test_x)

    # 'score' is the output of the perceptron
    def score(x, y):
        if x * y.T < 0:
            return -1
        else:
            return 1

    # Initialize the weight value
    weight = [0 for item in range(7)]
    # convert it to matrix
    weight = np.matrix(weight)
    # label is the evaluation of training group
    label = train_y.tolist()
    accuracy = 0
    # We set an accuracy threshold of 80%,
    # If the threshold is not reached, the entire training set is iterated repeatedly
    while accuracy < 0.8:
        count = 0
        for i in range(train_x.shape[0]):
            output = score(weight, train_x[i])
            if output != label[i]:  # Update the weight
                weight = weight + 0.1 * (label[i] - output) * train_x[i]
            else:
                count = count + 1
        accuracy = count / train_x.shape[0]
    # apply the weight to the test dataset
    predict = []
    for i in range(test_x.shape[0]):
        output = score(weight, test_x[i])
        predict.append(output)
    return predict


if __name__ == '__main__':
    # The following is the algorithm performance evaluation section
    print('first read the data set and preprocess it')
    # Preprocessing the data
    test = readCSV('test.csv')
    train_x, train_y, test_x, test_y = data_decomposition('training.csv')
    test_transform = data_transform(test)
    test_classifiers = ['KNN', 'Naive Bayes', 'Perceptron']
    classifiers = {'KNN': Knn,
                   'Naive Bayes': Naive_Bayes,
                   'Perceptron': Perceptron
                   }
    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        predict = classifiers[classifier](train_x, train_y, test_x)
        print('training took %fs!' % (time.time() - start_time))
        precision = metrics.precision_score(test_y, predict)
        recall = metrics.recall_score(test_y, predict)
        accuracy = metrics.accuracy_score(test_y, predict)
        f_score = metrics.f1_score(test_y, predict)
        print('precision: %.2f%%, recall: %.2f%%, F-score: %.2f%%' % (100 * precision, 100 * recall, 100 * f_score))
        print('accuracy: %.2f%%' % (100 * accuracy))
        print('take unacc as a positive example and acc as a negative example')
        test_y_reverse = [i * -1 for i in test_y]   # take unacc as a positive example and acc as a negative example
        predict_reverse = [i * -1 for i in predict]
        precision_reverse = metrics.precision_score(test_y_reverse, predict_reverse)
        recall_reverse = metrics.recall_score(test_y_reverse, predict_reverse)
        accuracy_reverse = metrics.accuracy_score(test_y_reverse, predict_reverse)
        f_score_reverse = metrics.f1_score(test_y_reverse, predict_reverse)
        print('precision: %.2f%%, recall: %.2f%%, F-score: %.2f%%' % (100 * precision_reverse
                                                                      , 100 * recall_reverse, 100 * f_score_reverse))
        print('accuracy: %.2f%%' % (100 * accuracy_reverse))

        # With the algorithm performance tested, we now predict the test set
        print('\nreading the test dataset and preprocess it ......')
        if not os.path.exists('test_predict_%s.csv' % classifier):
            train_x, train_y, test_x, test_y = data_decomposition('training.csv')
            ans = classifiers[classifier](train_x, train_y, test_transform)
            # the predicted values are converted to unacc and acc
            for i in range(len(ans)):
                if ans[i] == 1:
                    ans[i] = 'acc'
                else:
                    ans[i] = 'unacc'
            test_origin = readCSV('test.csv')
            # add the predict evaluation to the test.csv
            test_origin['pre_evaluation'] = ans
            test_origin.to_csv('test_predict_%s.csv' % classifier, index=False)
            print('%s algorithm application test dataset is created\n' % classifier)
        else:
            print('%s algorithm application test dataset is existed\n' % classifier)

