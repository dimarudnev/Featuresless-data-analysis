import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import glob
import random
import os 
import json
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, matthews_corrcoef, roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.manifold import MDS
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtw import dtw
import progressbar

from sklearn.semi_supervised import LabelSpreading
ROOT_DATA_PATH = r'C:\Data\hart-data-set\RawData'
ROOT_PATH = r'C:\Data\hart-data-set'

if __name__ == '__main__':
    def readSource(dataType, expId, userId):
        return pd.read_csv(os.path.join(ROOT_DATA_PATH, r'%s_exp%02d_user%02d.txt' % (dataType,expId,userId)), 
                           delimiter=" ", 
                           names=(dataType+"_X", dataType+"_Y", dataType+"_Z"))
    def plotAreas(ax, labels):
        for key, values in labels.iterrows():
            ax.text(0.5*(values['start']+values['end']), 1.5, values['act_id'],
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=30, 
                color='red')
            ax.axvspan(values['start'], values['end'], facecolor='#ccdfff', alpha=0.5)

    def plot_exp_series(expId, userId, labels):
        source_acc = readSource("acc", expId, userId)
        source_gyro = readSource("gyro", expId, userId) 

        exp_labels = labels.loc[expId, userId]



        f, (ax1, ax2) = plt.subplots(2)
        plotAreas(ax1, exp_labels)
        ax1.plot(source_acc)
        plotAreas(ax2, exp_labels)
        ax2.plot(source_gyro)



    def gererateData(labels):
        result = dict();
        for expId in labels.index.levels[0]:
        
            userId = labels.loc[expId].index.unique()[0]
    
            source = pd.concat([readSource("acc", expId, userId), readSource("gyro", expId, userId)], axis=1, join='inner')
            action_index = 0;
            for _, label, start, end in labels.loc[expId, userId].itertuples():
                key = "%02d-%02d-%02d" % (expId , userId, action_index);
            
                array = []
                values = source.loc[start:end].to_dict('list')
                action_index += 1
                result[key] = {
                    "label": str(label),
                    "values": values
                    }
                print(key)
        print("write results...")
        with open('hart_data.json', 'w') as fp:
            json.dump(result, fp)

    labelsDesc = {
        1 :"WALKING           ",
        2 :"WALKING_UPSTAIRS  ",
        3 :"WALKING_DOWNSTAIRS",
        4 :"SITTING           ",
        5 :"STANDING          ",
        6 :"LAYING            ",
        7 :"STAND_TO_SIT      ",
        8 :"SIT_TO_STAND      ",
        9 :"SIT_TO_LIE        ",
        10:"LIE_TO_SIT        ",
        11:"STAND_TO_LIE      ",
        12:"LIE_TO_STAND      "
    }
    def plot_basis(valuesArr, labelArr, rowCount=3, colCount=4):
        f, axarr = plt.subplots(rowCount, colCount)
        index = 0;
        for value in valuesArr:
            if index < rowCount * colCount:
                    row = index // colCount
                    col = index % colCount
                    axarr[row, col].plot(pd.DataFrame(value))
                    axarr[row, col].set_title("%s-%s"%( labelArr[index],labelsDesc[int(labelArr[index])]))
                    index+=1;

    def plot_label(data, label, colCount=5, rowCount=5):
        index = 0;
        f, axarr = plt.subplots(rowCount, colCount, sharex=True, sharey=True)
        f.suptitle('label %s (%s/%s)'% (label, rowCount, colCount))
        for k, value in data.items():
            if index < rowCount * colCount:
                if value['label']==label:
                    row = index // rowCount
                    col = index % colCount
                    axarr[row, col].plot(pd.DataFrame(value["values"]))
                    index+=1;
    #gererateData(labels)

    def randomBasis(values: [], labels: []):
        valuesRes = []
        labelsRes = []
        for i in range(0, 11):
            rnd = random.uniform(0, len(values))
            valuesRes.append(values[rnd]);
            labelsRes.append(labels[rnd]);

        return valuesRes, labelsRes

    def generateBasis(values, labels):
        classes = list(set(labels))
        res = [];
        resLabels = []
        for label_to_find in classes:
            for i, label in enumerate(labels):
                if label == label_to_find:
                    res.append(values[i])
                    resLabels.append(label)
                    break;
        return res, resLabels

    def calcSecondaryFeatures(basis, objects, dictanceFunc):
        pregressbar = progressbar.ProgressBar()
        return [[dictanceFunc(row, b) for b in basis] for row in pregressbar(objects)]
            

    def correlateDistance(object1: dict, object2: dict) -> int:
        return max(sum([np.correlate(object1[k], object2[k]) for k in object1.keys()]))
    def dtwDistance(object1: dict, object2: dict)-> int:
        return sum([fastdtw(object1[k], object2[k], dist=euclidean)[0] for k in object1.keys()])

    def plot_secondary_features(source_s, labels):
        print("----visualize----")
    
        plt.figure()
        scaled = MDS().fit_transform(source_s)
        def mapColor(label):
            return plt.cm.get_cmap('Paired')(float(label)/12);

        plt.scatter(scaled[:,0], scaled[:,1], c=list(map(mapColor, labels)), s=500, label=labels)
   

        plt.legend(handles = list([mpatches.Patch(color=mapColor(l), label="%s-%s"%(l,d)) for l, d in labelsDesc.items()]))


    labels = pd.DataFrame(pd.read_csv(r'C:\Data\hart-data-set\RawData\labels.txt',
     delimiter=" ",
     names=("exp_id", "user_id", "act_id", 'start', 'end'), 
     index_col=("exp_id", "user_id"),
     dtype ={"act_id": np.int}
     ))


    plot_exp_series(1,1, labels)
    
    
    def RandomForestClassifierWrapper(X_train, y_train, X_test):
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        return clf.predict(X_test)
    def LabelSpreadingWrapper(X_train, y_train, X_test):
        clf = LabelSpreading(
            kernel= 'knn',
            n_neighbors=10,
            n_jobs =-1,
            max_iter=1000,
            alpha=0.1
            )
        newlabels = np.concatenate((np.array(y_train), -np.ones(len(X_test))))
        clf.fit(np.concatenate((X_train, X_test)), newlabels);
        return clf.transduction_[-len(X_test):]
    #def CPLELearningWrapper(X_train, y_train, X_test):
    #    from frameworks.CPLELearning import CPLELearningModel
    #    #clf = RandomForestClassifier()
    #    from sklearn.linear_model.stochastic_gradient import SGDClassifier
    #    clf=  SGDClassifier(loss='log', penalty='l1')
    #    ssmodel = CPLELearningModel(clf)
    #    newlabels = np.concatenate((np.array(y_train), -np.ones(len(X_test))))
    #    ssmodel.fit(np.concatenate((X_train, X_test)), newlabels)
    #    return ssmodel.predict(X_test)
    def SelfTraingWrapper(X_train, y_train, X_test):
        from frameworks.SelfLearning import SelfLearningModel
        clf = RandomForestClassifier()   
        ssmodel = SelfLearningModel(clf)
        newlabels = np.concatenate((np.array(y_train), -np.ones(len(X_test))))
        ssmodel.fit(np.concatenate((X_train, X_test)), newlabels)
        return ssmodel.predict(X_test)

    def performTestSizeClassification(source_s, labels, classificationDict):
        plt.figure();
       
        for clfName, clf in classificationDict.items():
            results = [];
            print("\t classificator %s" % clfName)
            pregressbar = progressbar.ProgressBar()
            for testSize in pregressbar(np.arange(0.8, 0.99, 0.01)):
                results1 = [];
                for i in range(1, 2):
                    X_train, X_test, y_train, y_test = train_test_split(source_s, labels, test_size=testSize)
                    preds = clf(X_train, y_train, X_test)
                    results1.append(accuracy_score(preds, y_test))
                results.append(np.mean(results1))
            plt.plot(np.arange(0.8, 0.99, 0.01), results, label=clfName)
        plt.legend()
        #performDirectClassification();
                    
    def performDirectClassification(source_s, labels):
        results=[]
        sourceX = pd.DataFrame(pd.read_csv(os.path.join(ROOT_PATH, "Train", "X_train.txt"), delimiter=" ", header=None))
        sourceY = pd.DataFrame(pd.read_csv(os.path.join(ROOT_PATH, "Train", "y_train.txt"), delimiter=" ", header=None, names="Y"))['Y'].values
        
        for testSize in np.arange(0.8, 0.99, 0.01):
            results3 = []
            for i in range(1, 2):
                results3.append(directExpirement(sourceX, sourceY, testSize))
            results.append(np.mean(results3))
        plt.plot(np.arange(0.8, 0.99, 0.01), results)
    def performExpirement(title, 
                          values, 
                          labels, 
                          distanceFunc, 
                          generateBasisFunc,
                          classificationDict,
                          plot=False):
        print("-------------------------------------------")
        print("Start %s" % title)

        print("\t(%s) ----generate basis----" % title)
        basis, basisLabels = generateBasis(values, labels);
        if plot:
            plot_basis(basis, basisLabels)

        print("\t(%s) ----secondary features----" % title)
        source_s = np.array(calcSecondaryFeatures(basis, values, distanceFunc))
        if plot:
            plot_secondary_features(source_s, labels)

        print("\t(%s) ----classification----" % title)
        performTestSizeClassification(source_s, labels, classificationDict)
        acc = 0
        print("End %s" % title)
        return acc
    def directExpirement(sourceX, sourceY, testSize):
       
        X_train, X_test, y_train, y_test = train_test_split(sourceX, sourceY, test_size=testSize)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return accuracy_score(preds, y_test)
    with open('hart_data.json', 'r') as fp:

        print("----read data----")
        data = json.load(fp, object_pairs_hook=OrderedDict)
        labels = []
        values = []
        for k,v in data.items(): # dict does not garanture order of enumeration
            labels.append(int(v['label']))
            values.append(v['values'])


        performExpirement(
            title="Sample expirenemt",
            values= values, 
            labels= labels, 
            distanceFunc=correlateDistance, 
            generateBasisFunc= generateBasis,
            classificationDict= {
                #'CPLELearningWrapper': CPLELearningWrapper,
                'SelfTraingWrapper': SelfTraingWrapper,
                'RandomForestClassifierWrapper': RandomForestClassifierWrapper,
                'LabelSpreadingWrapper': LabelSpreadingWrapper
                },
            plot=False)

        #results1 = []
        #for i in range(0, 10):
        #    expirementRes = performExpirement("Different types %s" % i, values, labels, correlateDistance, generateBasis);
        #    results1.append(expirementRes)
        #results2= []
        #for i in range(0, 10):
        #    expirementRes = performExpirement("Random basis %s" % i, values, labels, correlateDistance, randomBasis);
        #    results2.append(expirementRes)
        
        #plt.figure()
        #plt.boxplot([results1, results2]);



    print("ready")
    plt.show()
