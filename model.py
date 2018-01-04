#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 00:12:55 2017
@author: chirag212

"""

# =============================================================================
# Dementia classification / MMSE prediction using LR, RF, DT or SVM
# =============================================================================

import os

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn import tree
from sklearn.svm import SVC
from scipy.stats import pearsonr as pearson
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score, confusion_matrix

seed = 212
def train(data, model):
    global args

    X = np.array([data['ttr'], data['R'], data['num_concepts_mentioned'], 
                  data['ARI'], data['CLI'], data['prp_count'], data['VP_count'], data['NP_count'], #data['DT_count'], 
                  data['prp_noun_ratio'], data['word_sentence_ratio'],
                  data['count_pauses'], data['count_unintelligible'], 
                  data['count_trailing'], data['count_repetitions']])
    
    X = X.T
    
    if args.type == 'DEM':
        y = np.array(data['Category']).T
    elif args.type == 'MMSE':
        y = np.array(data['MMSE'], dtype=np.int).T
        for i in range(len(y)):
            if y[i] in range(0, 21):
                y[i] = 0
            elif y[i] in range(21, 26):
                y[i] = 1
            elif y[i] in range(26, 31):
                y[i] = 2
            else:
                print('invalid MMSE')        
    else:
        print ('Invalid classification type')
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.20, random_state=212)
    train_samples, n_features = X.shape

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X_train, y_train):
        # Train the model
        if args.type=='DEM':
            if args.model == 'RF':
                model.n_estimators = 10
        elif args.type == 'MMSE':
            model.class_weight = {0:1, 1:2, 2:1}
        else:
            print ('Invalid classification type')        
        
        model.fit(X_train[train], y_train[train])    
        
        # evaluate the model
        y_pred = model.predict(X_train[test])
        
        # evaluate predictions
        cvscores.append(accuracy_score(y_train[test], y_pred))
#        print('Test accuracy for model: {}\n'.format(accuracy))
#        print ('F1-score: {}'.format(f1_score(y_train[test], y_pred, average=None)))
#        print ('Classification Report:\n')
#        print (classification_report(y_train[test], y_pred))
      
    print(np.mean(cvscores))
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Test accuracy for model: {}\n'.format(accuracy))
    print ('F1-score: {}'.format(f1_score(y_test, y_pred, average=None)))
    print (classification_report(y_test, y_pred))
  

#==============================================================================
#    Plotting PR- curves     
#==============================================================================
#    precision, recall, _ = precision_recall_curve(y_test, y_pred)
#    
#    plt.step(recall, precision, color='b', alpha=0.8,
#             where='post')
#    plt.fill_between(recall, precision, step='post', alpha=0.2,
#                     color='b')
#
#    #    plt.xticks(y_pos, temp_data[:, 0], fontsize=21.0, fontweight='bold', rotation='vertical')
#    plt.ylabel('Categorical count', fontsize=21.0, fontweight='bold')
#    plt.yticks(fontsize=21.0, fontweight='bold')
#
#    plt.xlabel('Recall', fontsize=21.0, fontweight='bold')
#    plt.ylabel('Precision', fontsize=21.0, fontweight='bold')
#    plt.ylim([0.0, 1.05])
#    plt.xlim([0.0, 1.0])
#    plt.yticks(fontsize=21.0, fontweight='bold')
#    plt.xticks(fontsize=21.0, fontweight='bold')
#    plt.title('2-class Precision-Recall curve')
    
def exploratory_analysis(data):
    box_features = ['ARI', 'CLI', 'count_trailing', 'count_repetitions', 'count_pauses', 'SIM_score', 'MMSE']
    for feat in box_features:
        temp_data = [np.array(data[feat][:242]), np.array(data[feat][data['Category']==1])]#, 
#                     np.array(data[feat][data['Category']==2]), np.array(data[feat][data['Category']==3])]

        plt.figure()
        plt.boxplot(temp_data, medianprops=dict(linestyle='-', linewidth=2, color='firebrick'))
        plt.xticks([1, 2], ['Control', 'AD'], fontsize=21.0, fontweight='bold')
        plt.yticks(fontsize=21.0, fontweight='bold')
#        plt.ylabel(feat, fontsize=21.0, fontweight='bold')
        plt.title(feat, fontsize=21.0, fontweight='bold')
        plt.show()
    
    box_features = ['ttr', 'R', 'num_concepts_mentioned', 
                  'ARI', 'CLI', 'prp_count', 'VP_count', 'NP_count',
                  'prp_noun_ratio', 'word_sentence_ratio',
                  'count_pauses', 'count_unintelligible', 
                  'count_trailing', 'count_repetitions']   
    for feat in box_features:
        [r, p] = pearson(data[feat], data['Category'])
        print ('{}--{}--{}'.format(feat, r**2, p))
        
def main():
    global args
    # ------- Input data from metadata.csv ------- 
    parser = argparse.ArgumentParser(description='Processing Dementia data')
    parser.add_argument('--file_path', default=os.path.join(os.getcwd(),'feature_set_dem.csv'), type=str,
                        help='filepath for Dementia classification feature set')    
#    parser.add_argument('--file_path', default=os.path.join(os.getcwd(),'feature_set_MMSE.csv'), type=str,
#                        help='filepath for MMSE prediction feature set')    
    parser.add_argument('--type', default='DEM', type=str,
                        help='type of classification DEM or MMSE')
    parser.add_argument('--model', default='LR', type=str,
                    help='model type')
        
    args = parser.parse_args()
    
    # Read data
    data = pd.read_csv(args.file_path, encoding='utf-8')
    
    # model
    if args.model == 'LR':
        model = LogisticRegression()
    elif args.model == 'DT':
        model = tree.DecisionTreeClassifier()
    elif args.model == 'SVM':
        model = SVC()
    elif args.model == 'RF':
        model = RandomForestClassifier()
    else:
        print ('Invalid model')
        
    train(data, model)
            
if __name__ == '__main__':    
    main()