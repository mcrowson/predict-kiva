# -*- coding: utf-8 -*-
'''
Created on Mon Jul  7 12:46:25 2014

The following is nice for saving and loading models
>>> from sklearn.externals import joblib
>>> joblib.dump(clf, 'my_model.pkl', compress=9)
And then later, on the prediction server:

>>> from sklearn.externals import joblib
>>> model_clone = joblib.load('my_model.pkl')


@author: matthew
'''

import pymongo
import mongodb_proxy
import logging
import pylab as pl
import pandas as pd
from sklearn import linear_model, grid_search, svm
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, f1_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.feature_extraction import text
from sklearn.pipeline import Pipeline
import matplotlib.pylab as plt
import numpy as np
import multiprocessing as mp
import os
import random as ran
import sys
from operator import sub

#Script Logging
LEVEL = logging.DEBUG
FORMAT = logging.Formatter('%(asctime)-15s %(name)s %(levelname)-8s %(message)s')
log = logging.getLogger(__name__)
log.setLevel(LEVEL)
fhandler = logging.FileHandler('modeling.log')
shandler = logging.StreamHandler()
fhandler.setFormatter(FORMAT)
shandler.setFormatter(FORMAT)
log.addHandler(fhandler)
log.addHandler(shandler)

log.info('Starting Modeling Script')


class MongoConnection():
    '''Creates an instance of a connection to the mongo DB'''
    def __init__(self):
        self._uri = 'mongodb://app:3knvak3ijs@localhost/kiva'
        try:
            self.client = mongodb_proxy.MongoProxy(pymongo.MongoClient(self._uri))        
            self.db = self.client['kiva']
        except:
            log.error('Could not establish a connection to Mongo Client')


def sizeof_fmt(num):
    '''Returns the size of an object in human readable format'''
    for x in ['bytes', 'KB', 'MB', 'GB']:
        if -1024.0 < num < 1024.0:
            return "%3.1f%s" % (num, x)
        num /= 1024.0
    return "%3.1f%s" % (num, 'TB')
            
if __name__ == '__main__':
    #Make Directories We will use
    directories = ['figs', 'figs/results']
    for folder in directories:
        if not os.path.isdir(str(os.sep).join([os.getcwd(), folder])):
            os.mkdir(str(os.sep).join([os.getcwd(), folder]))
    # Connect to Mongo
    mongo_conn = MongoConnection()
    kiva_db = mongo_conn.db


    flat_loan_collection = kiva_db['flat_loans']

    #Get a random sample of loans
    np.random.seed(8798)
    threshold = np.random.random()
    log.debug('Random value is %s' % threshold)
    # Of the 570,343 samples we will use 20%
    sample_size = int(1 * 570343)
    c = 0
    while c < sample_size:  # Go until we have enough observations for the sample size
        cursor = flat_loan_collection.find({},  # dict(random={'$gte': threshold})
                                           dict(_id=0, journal_totals_bulkEntries=0, journal_totals_entries=0,
                                                translator_byline=0, translator_image=0, video_title=0, video_id=0,
                                                image_template_id=0, image_id=0, paid_amount=0, random=0,
                                                video_thumbnailImageId=0, video_youtubeId=0, final_pmt_days_late=0,
                                                delinquency_decile_0=0, delinquency_decile_1=0, delinquency_decile_10=0,
                                                delinquency_decile_2=0, delinquency_decile_3=0, delinquency_decile_4=0,
                                                delinquency_decile_5=0, delinquency_decile_6=0, delinquency_decile_7=0,
                                                delinquency_decile_8=0, delinquency_decile_9=0,
                                                currency_exchange_loss_amount=0, actual_days_to_pay=0))
        c = cursor.count()
        c = sample_size # temporary to get all loans
    loans = [l for l in cursor[:int(sample_size)]]
    log.debug(u'The loan collection is taking up {0:s} space'.format(sizeof_fmt(sys.getsizeof(loans))))
    loans = pd.DataFrame(loans)
    log.debug('The data set contains %(size)i observations with %(def)i defaulted loans'
              % {'size': len(loans), 'def': len(loans.ix[loans['defaulted'] == 1])})

    del_loans = loans.ix[loans['defaulted'] == 0]  # Delinquent Loan Set
    def_loans = loans.drop('dollar_days_late_metric', 1)  # Defaulted Loan Set
    loans = None
    log.debug('Dataframe split into 2 sub dataframes')

    #Determine the text fields
    text_fields = [col for col in del_loans.columns.values.tolist() if 'description_texts' in col]
    text_fields += [u'id', u'use', u'activity']
    text_df = del_loans.ix[:, text_fields]
    [del_loans.drop(var, 1, inplace=True) for var in text_fields if var != 'id']


    #Train/Test Split for Delinquent Loans - May want to use cross validation later
    X = del_loans.drop('dollar_days_late_metric', 1)
    Y = del_loans['dollar_days_late_metric']

    # Train / Test Split 70% / 30%
    train_mask = np.random.random(len(X)) <= .7
    train_x, train_y = X[train_mask], Y[train_mask]
    test_x, test_y = X[~train_mask], Y[~train_mask]


    train_text_df = text_df[train_mask]
    test_text_df = text_df[~train_mask]


    # Remove rows with non-numeric data in any field. This hopefully has no effect.
    train_x = train_x[train_x.applymap(lambda x: isinstance(x, (int, float))).all(1)].fillna(value=0)
    test_x = test_x[test_x.applymap(lambda x: isinstance(x, (int, float))).all(1)].fillna(value=0)
    log.debug('Testing and Training data have removed all non int and float rows.')

    '''
    I'll need to pickle the transformers for future test data.
    '''

    '''
    # Vectorize text fields and add them back to our training and testing dfs
    for field in text_fields:  # Remove the [:1] when done with testing
        if field not in [u'activity', u'use']: #We are ignoring description texts for now due to high RAM requirements
            continue
        log.debug('Creating text vectors for %s' % field)
        vect = text.TfidfVectorizer()

        train_data = vect.fit_transform(train_text_df[field].fillna('')).toarray()
        train_df = pd.DataFrame(data=train_data,
                                columns=['_'.join([field, name]) for name in vect.get_feature_names()])
        train_x = train_x.join(train_df).fillna(value=0)
        train_text_df.drop(field, 1, inplace=True)

        test_data = vect.transform(test_text_df[field].fillna('')).toarray()
        test_df = pd.DataFrame(data=test_data,
                               columns=['_'.join([field, name]) for name in vect.get_feature_names()])
        test_x = test_x.join(test_df).fillna(value=0)
        test_text_df.drop(field, 1, inplace=True)

    log.debug('Text variables vectorized and applied to dataframes.')
    '''

    #test_x.drop('id', 1, inplace=True)
    #train_x.drop('id', 1, inplace=True)

    #Model Creations If there are parameters set in the grid, they were done so with Cross Validation.
    reg_models = [#{'name': 'Linear Regression',
               #'object': linear_model.LinearRegression()},
              #{'name': 'Lasso',
              # 'object': linear_model.Lasso(alpha=.0041)}, # Best alpha = 0.0041
              #{'name': 'Nearest Neighbors Regression',
              # 'object': KNeighborsRegressor( n_neighbors=9, p=2, weights='uniform')}, # 9 is Best between 1 and 100
              {'name': 'Random Forest Regressor',
               'object': RandomForestRegressor(max_depth=None, max_features=0.3, n_estimators=200, n_jobs=-1),
               'grid': {'min_samples_leaf': range(1, 8, 2)}} #  n_estimators=170, max_features=.3,min_samples_leaf=5,
               ]

    #Models had the following R^2 scores on test data
    # Linear Regression: 0.570571955820
    # Lasso :            0.565441691494
    # KNN:               0.267601452273
    # Random Forest:     0.720037591000
    for model in reg_models:
        reg = model['object']
        #Grid Search for Parameters if Defined in Models Array

        grid = model.get('grid', None)
        if grid:
            log.debug('Running grid search for optimal settings via 10-fold CV')
            reg = grid_search.GridSearchCV(reg, model['grid'], n_jobs=-1, cv=10)
            reg.fit(train_x, train_y)
            for params, mean_score, scores in reg.grid_scores_:
                log.debug("%0.3f (+/-%0.03f) for %r" % (
                    mean_score, scores.std() / 2, params))
            
            log.info(reg.best_estimator_)
        else:
            reg.fit(train_x, train_y)
        
        predictions = reg.predict(test_x)
        train_score = r2_score(train_y, reg.predict(train_x))
        test_score = r2_score(test_y, predictions)
        log.debug(reg.get_params())
        log.info(' '.join([model['name'], 'Performance']))
        log.info(' '.join(['Training Data', str(train_score)]))
        log.info(' '.join(['Test Data', str(test_score)]))


        if model['name'] == 'Random Forest Regressor':
            importances = reg.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Print the feature ranking
            log.debug("Feature ranking:")

            for f in range(25):
                log.info("%d. %s (%f)" % (f + 1, train_x.columns[indices[f]], importances[indices[f]]))


        '''
        #Plot the Predicted vs Actual
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
        plt.scatter(test_y[::50], predictions[::50], alpha=.25)  # Stepping 50 so sample plotting
        plt.title(model['name'])
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        textstr = '$R^{2}$=%.3f' % test_score
        props = dict(boxstyle='round', facecolor='white')
        ax.text(1.02, 0.92, textstr, fontsize=14, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
        plt.grid(False)
        fig.savefig('figs/results/%s.png' % model['name'])


        #Plot the Residuals
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
        residuals = map(sub, test_y[::50], predictions[::50])
        plt.scatter(test_y[::50], residuals, alpha=.25)  # Stepping 50 so plotting doesn't take so long
        plt.title('%s Residuals' % model['name'])
        plt.xlabel('X Value')
        plt.ylabel('Residual')
        textstr = '$R^{2}$=%.3f' % test_score
        props = dict(boxstyle='round', facecolor='white')
        ax.text(1.02, 0.92, textstr, fontsize=14, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
        plt.grid(False)
        fig.savefig('figs/results/%s_residuals.png' % model['name'])


        #Plot the Score as we add observations. Useful for identifying bias and variance, but takes a added time.
        train_r2 = []
        test_r2 = []
        examples = xrange(1000, sample_size, int(sample_size/10))
        for i in examples:
            sub_train_x = train_x[:i]
            sub_train_y = train_y[:i]        
            reg.fit(sub_train_x, sub_train_y)
            sub_train_predictions = reg.predict(sub_train_x)
            sub_train_score = r2_score(sub_train_y, sub_train_predictions)
            train_r2.append(sub_train_score)
            predicti\
                    ons = reg.predict(test_x)
            score = r2_score(test_y, predictions)
            test_r2.append(score)
        
        plt.figure()
        plt.plot(examples, train_r2, 'r-', label='Train')
        plt.plot(examples, test_r2, 'b-', label='Test')
        plt.legend()
        plt.grid(False)
        plt.savefig('figs/results/%s_train_test' % model['name'])
        '''
    '''

    #Model Creations If there are parameters set in the grid, they were done so with Cross Validation.
    log.info('Beginning Default Classifier Modeling')


    #Remove the text fields
    [def_loans.drop(var, 1, inplace=True) for var in text_fields if var != 'id']

    #Train/Test Split for Delinquent Loans - May want to use cross validation later
    X = def_loans.drop('defaulted', 1)
    Y = def_loans['defaulted']
    def_loans = None

    # Train / Test Split 70% / 30%
    train_mask = np.random.random(len(X)) <= .7
    train_x, train_y = X[train_mask], Y[train_mask]
    test_x, test_y = X[~train_mask], Y[~train_mask]
    X, Y = None, None
    log.debug('The training set include %i defaulted loans and the test set includes %i defaulted loans'
              % (sum(train_y), sum(test_y)))

    # Remove rows with non-numeric data in any rogue field.
    train_x = train_x[train_x.applymap(lambda x: isinstance(x, (int, float))).all(1)].fillna(value=0)
    test_x = test_x[test_x.applymap(lambda x: isinstance(x, (int, float))).all(1)].fillna(value=0)
    log.debug('Testing and Training data have removed all non int and float rows.')


    test_x.drop('id', 1, inplace=True)
    train_x.drop('id', 1, inplace=True)

    classifier_models = [
              #{'name': 'Nearest Neighbors Classifier',
              # 'object': KNeighborsClassifier()},
              {'name': 'Random Forest Classifier',
               'object': RandomForestClassifier(n_jobs=-1)},
              #{'name': 'Guasian Naive Bayes', 'object': GaussianNB()}
               ]

    #Create Default Models
    for model in classifier_models:
        clf = model['object']
        #Grid Search for Parameters if Defined in Models Array

        grid = model.get('grid', None)
        if grid:
            clf = grid_search.GridSearchCV(clf, model['grid'], n_jobs=-1, cv=10)
            clf.fit(train_x, train_y)
            for params, mean_score, scores in clf.grid_scores_:
                print "%0.3f (+/-%0.03f) for %r" % (
                    mean_score, scores.std() / 2, params)

            log.info(clf.best_estimator_)
        else:
            clf.fit(train_x, train_y)

        predictions = clf.predict(test_x)
        train_score = accuracy_score(train_y, clf.predict(train_x))
        test_score = accuracy_score(test_y, predictions)
        roc_score = roc_auc_score(test_y, predictions)
        log.debug(clf.get_params())
        log.info(' '.join([model['name'], 'Performance']))
        log.info(' '.join(['Training Data', str(train_score)]))
        log.info(' '.join(['Test Data', str(test_score)]))

        if model['name'] == 'Random Forest Classifier':
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Print the feature ranking
            log.debug("Feature ranking:")

            for f in range(25):
                log.info("%d. %s (%f)" % (f + 1, train_x.columns[indices[f]], importances[indices[f]]))


        #Plot the Predicted vs Actual
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])
        plt.scatter(test_y[::50], predictions[::50], alpha=.25)  # Stepping 50 so sample plotting
        plt.title(model['name'])
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        textstr = '$F$=%.3f' % test_score
        props = dict(boxstyle='round', facecolor='white')
        ax.text(1.02, 0.92, textstr, fontsize=14, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
        plt.grid(False)
        fig.savefig('figs/results/%s.png' % model['name'])


        # Plot ROC curve
        pl.clf()
        fpr, tpr, thresholds = roc_curve(test_y, clf.predict_proba(test_x)[:, 1])
        pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_score)
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic example')
        pl.legend(loc="lower right")
        pl.savefig('figs/results/roc_%s.png' % model['name'])
    '''

shandler.close()
fhandler.close()

'''
Pickle and save the model to file.
'''