from sklearn.externals import joblib
import numpy as np
from sklearn.cluster import KMeans
import hmmlearn.hmm as hmm
from numpy import inf

numFeature = 15
# filename = 'database_train.pkl'
# filename = 'database_test.pkl'
def getTest(filename):
    data = joblib.load(filename)
    numObs = int(data.shape[1] / numFeature)
    # data.shape = (1, numObs * 15)
    XTest = data.reshape((numObs, numFeature))
    return XTest
    
def loadData(filename):
    data = joblib.load(filename)
    XTrain = dict()
    numRec = dict()
    for key in data.keys():
        if (key != 'recNum'):
            data[key] = np.array(data[key])
            # data[key].shape is (numRec, n*15)
            print(data[key].shape)
            numRec[key] = data[key].shape[0]
            numObs = int(data[key].shape[1] / numFeature)
            data[key] = np.array(data[key]).reshape((numRec[key] * numObs, numFeature))
            XTrain[key] = data[key]
    # return dictionary XTrain, XTrain['TK'].shape (numRec * n, 15)
    return XTrain, numRec, numObs

# input: XTrain: dict(); numRec: dict(); numObs: int; numCluster: int.
def vectorQuant(XTrain, numRec, numObs, numCluster = 10):
    kmeans_model = dict()
    OTrain = dict()
    for key in XTrain:
        X = XTrain[key] # Shape: (numRec * n, 15)
        kmeans_model[key] = KMeans(numCluster).fit(X)
        OTrain[key] = kmeans_model[key].labels_.reshape((numRec[key], numObs))
        #values = kmeans_model[key].cluster_centers_.squeeze()
    # return: kmeans_model:the dict store kmeans models; the model can be used in test set.
    # return: OTrain: the dict store the converted obs for each rec event, OTrain[key].shape = (numRec[key], numObs)
    return kmeans_model, OTrain
# kmeans_model_RK.predict(XTest)

# input: the converted training set OTrain, num of states, num of iteration.
# return: the dict() store all the trained hmm model.
def trainHMM(OTrain, numSts, numIter):
    hmm_model = dict()
    isConverged = False
    numFit = 3
    currScore = -inf
    maxScore = -inf
    currModel = None
    bestModel = None
    for key in OTrain.keys():
        # iterate the fitting model untill converge.
        for i in range(numFit):
            while (True):
                currModel = hmm.MultinomialHMM(n_components = numSts, n_iter = numIter).fit(OTrain[key])
                currScore = currModel.score(OTrain[key])
                if (currModel.monitor_.converged):
                    break
            if currScore >= maxScore:
                maxScore = currScore
                bestModel = currModel
        hmm_model[key] = bestModel
    return hmm_model

# input: one record of technique, XTest, shape: (numObs*15)
def predict(XTest, numObs, kmeans_model, hmm_model):
    log_l = dict()
    for key in hmm_model.keys():
        # first convet x to obs
        XTest = XTest.reshape(numObs, numFeature)
        OTest = kmeans_model[key].predict(XTest).reshape((1, numObs)) # OTest.shape = (1, numObs)
        log_l[key] = hmm_model[key].score(OTest) # a float number : lob_prob
    return log_l

# this run func is arbitrarily designed for debugging purpose.
def run_debug():
    XTrain, numRec, numObs = loadData('database_test.pkl')
    XTest, numRec_test, numObs_test = loadData('database_train.pkl')
    kmeans_model, OTrain = vectorQuant(XTrain, numRec, numObs, numCluster = 10)
    numSts = 6
    numIter = 50
    hmm_model = trainHMM(OTrain, numSts, numIter)
    for key in XTest.keys():
        xTest = XTest[key].reshape((numRec_test[key], numObs * numFeature))[9]
        log_l = predict(xTest, numObs, kmeans_model, hmm_model)
        print(key, log_l)

# press train button will call the func
def train(filename):
    XTrain, numRec, numObs = loadData(filename)
    kmeans_model, OTrain = vectorQuant(XTrain, numRec, numObs, numCluster = 10)
    numSts = 6
    numIter = 50
    hmm_model = trainHMM(OTrain, numSts, numIter)
    models = (numObs, kmeans_model, hmm_model)
    joblib.dump(models, 'model.pkl')

 # press test button will call the func
def test(xTest, filename):
    (numObs, kmeans_model, hmm_model) = joblib.load(filename)
    log_l = predict(xTest, numObs, kmeans_model, hmm_model)
    output = None
    min = inf
    for key in log_l.keys():
        if log_l[key] < min:
            output = key
            min = log_l[key]
    if output == 'MP':
        output = 'Middle Punch'
    else:
        output = 'Round Kick'
    return output # a string
    

