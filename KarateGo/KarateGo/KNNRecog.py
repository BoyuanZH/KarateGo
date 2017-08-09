from sklearn.externals import joblib
import numpy as np
from numpy import inf
import copy
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

n_components = 1
numFeature = 12
frameRate = 60
code_book = ['Left Round Kick', 'Right Round Kick', 'Left Middle Punch', 'Right Middle Punch']
code_book_dict = {'LRK': 0, 'RRK': 1, 'LMP':2, 'RMP':3}

def loadData(filename):
    data = joblib.load(filename)
    XTrain = dict()
    numRec = dict()
    for key in data.keys():
        if (key != 'recNum'):
            data[key] = np.array(data[key])
            # data[key].shape is (numRec, n*15)
            numRec[key] = data[key].shape[0]
            numObs = int(data[key].shape[1] / numFeature)
            data[key] = np.array(data[key]).reshape((numRec[key] * numObs, numFeature))
            XTrain[key] = data[key]
    # return dictionary XTrain, XTrain['TK'].shape (numRec * n, 15)
    return XTrain, numRec, numObs

def train(data_filename, model_filename):
    XTrain, numRec, numObs = loadData(data_filename)
    Train = None
    numTpye = len(XTrain)
    for key in XTrain.keys():
        newKey = code_book_dict[key]
        numRec = int(XTrain[key].shape[0] / frameRate)
        Y = np.ones((numRec, 1)) * newKey
        X = XTrain[key].reshape(numRec, 60 * 12)
        YX = np.hstack((Y, X))
        if Train is None:
            Train = copy.deepcopy(YX)
        else:
            Train = np.vstack((Train, YX))

    XTrain= Train[:, 1:]
    YTrain = Train[:, 0]
    # apply pca
    pca = PCA(n_components = n_components) # 1
    XTrain = pca.fit_transform(XTrain)
    # fit a knn model
    clf = KNeighborsClassifier(n_neighbors=numTpye)
    clf.fit(XTrain, YTrain)
    models = (pca, clf)
    joblib.dump(models, model_filename) #'knn_model.pkl'
    return models

# XTest.shape = (1, frameRate * numFeature,)
def predict(XTest, model_filename):
    (pca, clf) = joblib.load(model_filename)
    XTest = XTest.reshape(1, -1)
    XTest = pca.transform(XTest)
    pred = int(clf.predict(XTest)[0])
    pred = code_book[pred]
    return pred

def predict_real_time(XTest, models):
    (pca, clf) = models
    XTest = XTest.reshape(1, -1)
    XTest = pca.transform(XTest)
    pred = int(clf.predict(XTest)[0])
    pred = code_book[pred]
    return pred

def testKmeansAccuracy():
    XTrain, numRec, numObs = loadData('database.pkl')
    Train = None
    Test = None
    for key in XTrain.keys():
        newKey = code_book_dict[key]
        numRec = int(XTrain[key].shape[0] / 60)

        numTrain = 8
        numTest = numRec - numTrain
        X = XTrain[key].reshape(numRec, 60 * 12)
        Y = np.ones((numRec, 1)) * newKey
        YX = np.hstack((Y, X))

        yxtrain = YX[:numTrain]
        yxtest = YX[numTrain:]
        if Train is None:
            Train = copy.deepcopy(yxtrain)
        else:
            Train = np.vstack((Train, yxtrain))

        if Test is None:
            Test = copy.deepcopy(yxtest)
        else:
            Test = np.vstack((Test, yxtest))

    # print(Train.shape) # (48, 721)
    # print(Test.shape) #(23, 721)
    # np.random.shuffle(Train)
    # np.random.shuffle(Test)

    XTrain = Train[:, 1:]
    YTrain = Train[:, 0]
    XTest= Test[:, 1:]
    YTest = Test[:, 0]

    # apply pca
    pca = PCA(n_components = n_components)
    XTrain = pca.fit_transform(XTrain)


    x = XTest[15].reshape(1, -1)

    x = pca.transform(x)
    XTest = pca.transform(XTest)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(XTrain, YTrain)
    test_accuracy = clf.score(XTest, YTest)


    print(clf.predict(x))
    print('predictions:')
    print(clf.predict(XTest))
    print('true values:')
    print(YTest)
    print(test_accuracy)


# XTest, YTest = getSampe()
# print(YTest)
# train(data_filename = "database.pkl", model_filename = "knn_model.pkl")
# pred = predict(XTest, model_filename = "knn_model.pkl")
# print(pred)
