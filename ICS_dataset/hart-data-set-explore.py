import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import glob
import os 
import json
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, matthews_corrcoef, roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.manifold import MDS
ROOT_DATA_PATH = r'C:\Data\hart-data-set\RawData'



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


def generateBasis(data: dict):
    classes = list(map(str, range(1, 13)))
    res = [];
    resLabels = []
    for label_to_find in classes:
        for key, value in data.items():
            if value['label'] == label_to_find:
                res.append(value['values'])
                resLabels.append(value['label'])
                break;
    return res, resLabels

def calcSecondaryFeatures(basis, objects, dictanceFunc):
    return [[dictanceFunc(row, b) for b in basis] for row in objects]

def correlateDistance(object1: dict, object2: dict) -> int:
    return max(sum([np.correlate(object1[k], object2[k]) for k in object1.keys()]))


labels = pd.DataFrame(pd.read_csv(r'C:\Data\hart-data-set\RawData\labels.txt',
 delimiter=" ",
 names=("exp_id", "user_id", "act_id", 'start', 'end'), 
 index_col=("exp_id", "user_id"),
 dtype ={"act_id": np.int}
 ))


plot_exp_series(1,1, labels)
    

with open('hart_data.json', 'r') as fp:

    print("----read data----")
    data = json.load(fp, object_pairs_hook=OrderedDict)

    print("----generate basis----")
    basis, basisLabels = generateBasis(data);
    plot_basis(basis, basisLabels)

    labels = []
    values = []
    for k,v in data.items():
        labels.append(int(v['label']))
        values.append(v['values'])
    
    print("----secondary features----")
    source_s = np.array(calcSecondaryFeatures(basis, values, correlateDistance))
    trainX, testX, trainY, testY = train_test_split(source_s, labels, train_size = .1)
    
    print("----visualize----")
    
    plt.figure()
    scaled = MDS().fit_transform(source_s)
    def mapColor(label):
        return plt.cm.get_cmap('Paired')(float(label)/12);

    plt.scatter(scaled[:,0], scaled[:,1], c=list(map(mapColor, labels)), s=500, label=labels)
   

    plt.legend(handles = list([mpatches.Patch(color=mapColor(l), label="%s-%s"%(l,d)) for l, d in labelsDesc.items()]))

    print("----classification----")
    clf = RandomForestClassifier()
    clf.fit(trainX, trainY)
    preds = clf.predict(testX)

    print("accuracy_score: {:.3f}".format(accuracy_score(testY, preds)))
    #print("roc_auc_score: {:.3f}".format(roc_auc_score(testY, preds)))
    #print("mcc: {:.3f}".format(matthews_corrcoef(testY, preds)))
    #print("f1: {:.3f}".format(f1_score(testY, preds)))
    print("Confusion matrix:\n%s" % confusion_matrix(testY, preds))

print("ready")
plt.show()
