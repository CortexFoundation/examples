#coding=utf-8

import os
import json
import codecs
import glob

import h5py
import numpy as np
import pandas as pd
import bottleneck

# 1. Consider Order Of Class Directories. Pytorch use alphabetical instead of info in json
class_label_alphabet = [item.split('/')[-1] for item in glob.glob('/data3/inat_reorder/train/*')]
class_label_alphabet.sort()
class_label_alphabet = [ item.decode('utf-8').encode('utf-8') for item in class_label_alphabet]
class_label_alphabetDF = pd.DataFrame(class_label_alphabet)
class_label_alphabetDF = class_label_alphabetDF.reset_index()
class_label_alphabetDF.columns = ['originID','fullname']

# 2. Read true order of classes
with codecs.open('runs_iNat/train2017.json',encoding='utf-8') as f:
    traindict = json.load(f)
    
trainIDDF = pd.DataFrame(traindict['categories'])

trainIDDF = trainIDDF.sort_values(['supercategory','name'])

LabelIDs = pd.concat([class_label_alphabetDF, trainIDDF.reset_index()], axis=1)
LabelIDs = LabelIDs.sort_values('fullname').reset_index()
ConvertIDDict = LabelIDs['id'].to_dict()

print ConvertIDDict[0]


def convertID(s,cutoff=5):
    return ' '.join([str(ConvertIDDict[item]) for item in list(bottleneck.argpartition(-1*s.values,cutoff)[:cutoff])])

# 3. Read true test pic id without classes

with codecs.open('runs_iNat/test2017.json') as f:
    sampleDict = json.load(f)
    
sampleDF = pd.DataFrame(sampleDict['images'])
sampleDF = sampleDF.sort_values('file_name')
sampleDF = sampleDF.reset_index()
sampleDF.index = sampleDF['file_name']

sampleDict = sampleDF['id'].to_dict()


NamerID = pd.DataFrame(zip([item.split('/')[-1] for item in glob.glob('/data4/test2017cvpr/val/val000/*.jpg')],range(182707)))
#zip(glob.glob('/data4/test2017cvpr/val/val000/*.jpg'),range(182707))
NamerID.columns = ['Namer','NamerID']
NamerDict = NamerID['Namer']



# 4. Define Where To Read/Save
corefix = 'resnext38_16x32d1ov2p3wd0nes9last'
prefix = '/data4/runs_iNat/{0}/'.format(corefix)
resfilename = 'Result_0_{0}.hdf'

for i in range(1,7):
    print i,
    tmpDF = np.exp(pd.read_hdf(prefix+resfilename.format(i),'result').values)
    tmpCleanDF = np.diag(1.0/tmpDF.sum(axis=1)).dot(tmpDF)
    if i == 1 :
        CleanDF = tmpCleanDF
    else:
        CleanDF = CleanDF + tmpCleanDF
    DF = pd.DataFrame(CleanDF)
    DF['true_id'] = DF.index.map(lambda x: sampleDict[NamerDict[x]] )
    DF = DF.sort_values('true_id')
    (DF.iloc[:,:-1]).to_hdf(prefix + 'Result_run0_MultiCenterCrop_1to{0}.hdf'.format(i),'result')
    X = DF.iloc[:,0:5089].apply(convertID,axis=1)
    submission = pd.DataFrame(X).reset_index().reset_index()
    submission.columns = ['id','hddid','predicted']
    submission[['id','predicted']].to_csv(\
           prefix + 'Submission_{0}_MultiCenterCrop_1to{1}.csv'.format(corefix,i),header=True,index=False,sep=',',encoding='utf-8')


    






