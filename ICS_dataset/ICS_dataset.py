import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from hyperopt import hp, fmin, Trials, tpe
from sklearn.metrics import make_scorer, matthews_corrcoef, roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn import svm
from FastMap import FastMap
from scipy.spatial import distance
from itertools import product
import time
import multiprocessing
import threading
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from functools import reduce

datadir = 'C:\\Data\\ics-data-set\\binaryAllNaturalPlusNormalVsAttacks'

if __name__ == '__main__':
    def readData(fileIndex):
        return pd.read_csv(os.path.join(datadir, 'data%d.csv' % fileIndex))

    def searchOptimalParameters(dtrain):
        def score(params):
            params['silent'] = 1
            params['num_class'] =  2
            acc = xgb.cv(params, dtrain)['test-merror-mean'].mean()
            print(acc)
            return acc

        space = {
            'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
            'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
            'max_depth' : hp.choice('max_depth', np.arange(1, 14, dtype=int)),
            'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
            'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
            'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
            'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05)
        }

        trials = Trials()
        return fmin(score, space, algo=tpe.suggest, max_evals=100, trials=trials)

    def performClassficationTest(trainX, testX, trainY, testY):
        dtrain = xgb.DMatrix(trainX, label=trainY)
        dvalid = xgb.DMatrix(testX, label=testY)

        #bestParams = searchOptimalParameters(dtrain)
        bestParams = {'n_estimators': 790.0, 'min_child_weight': 2.0, 'subsample': 1.0, 'colsample_bytree': 0.55, 'eta': 0.225, 'max_depth': 11, 'gamma': 0.6000000000000001}

        bestParams['silent'] = 1
        bestParams['num_class'] =  2
        bestClf = xgb.train(bestParams, dtrain)
        preds = bestClf.predict(dvalid)
        printResults(testY, preds)
        return roc_auc_score(testY, preds)

    def printResults(testY, preds):
        return
        #print("accuracy_score: {:.3f}".format(accuracy_score(testY, preds)))
        #print("roc_auc_score: {:.3f}".format(roc_auc_score(testY, preds)))
        #print("mcc: {:.3f}".format(matthews_corrcoef(testY, preds)))
        #print("f1: {:.3f}".format(f1_score(testY, preds)))
        #print("Confusion matrix:\n%s" % confusion_matrix(testY, preds))
    
    def generateBasis(basisCount, trainX):
        kmeans = KMeans(n_clusters=basisCount, random_state=0, n_jobs=multiprocessing.cpu_count()).fit(trainX)
        return kmeans.cluster_centers_

    def calcSecondaryFeatures(dataFrame, basis, functions):
        values = dataFrame.values
        if functions is not None:
            for f in functions:
                values = f(values)
        

        return [[distance.euclidean(row, b) for b in basis] for row in values]
        

    def createScaler(trainX):
        scaler = RobustScaler()
        scaler.fit(trainX)
        
        return lambda array: scaler.transform(array)
    def createImportanceScaler(trainX, trainY):
        forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
        forest.fit(trainX, trainY)
        return lambda array: array * forest.feature_importances_ 

    source = readData(1)
    for i in range(2, 15):
        source = source.append(readData(i), ignore_index=True)

    source = source.replace(np.inf, np.nan).fillna(0)
    elementNames = ['R1', 'R2', 'R3', 'R4']

    sourceY = [1 if marker== 'Attack' else 0 for marker in source['marker']]
    sourceX = source.drop(['marker'], axis=1)

    trainX, testX, trainY, testY = train_test_split(sourceX, sourceY, train_size = 2000, test_size= 20000)



    print("-----   Direct Expirement   ----")
    target = performClassficationTest(trainX, testX, trainY, testY)

    scaleFunc = createScaler(trainX)
    importanceFunc = createImportanceScaler(trainX, trainY)


    res = []
    


    for basisCount in range(5,15):
        roundres= []
        for round in range(0, 10):
            trainX, testX, trainY, testY = train_test_split(sourceX, sourceY, train_size = 2000, test_size= 20000)
            print("-----   Expirement basis %s / round %s  ----" % (basisCount, round))
            basis = generateBasis(basisCount, trainX)

            trainX_s = calcSecondaryFeatures(trainX, basis, [scaleFunc, importanceFunc])
            testX_s = calcSecondaryFeatures(testX, basis, [scaleFunc, importanceFunc])
            roundres.append(performClassficationTest(trainX_s, testX_s, trainY, testY))
        res.append(roundres)




    plt.plot([target]*10)
    plt.boxplot(res)
    plt.xticks(range(5,15))
    plt.show()


   





