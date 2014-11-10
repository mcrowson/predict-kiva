# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 08:17:35 2013

Exploratory Data Analysis for Kiva Data

@author: Matthew Crowson
"""

import pymongo
from utils import mongodb_proxy
import logging
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, f_oneway
import multiprocessing as mp
import numpy as np
import os
import math

#Script Logging
LEVEL = logging.DEBUG
FORMAT = logging.Formatter('%(asctime)-15s Line %(lineno)s %(name)s %(levelname)-8s %(message)s')
log = logging.getLogger(__name__)
log.setLevel(LEVEL)
fhandler = logging.FileHandler('eda.log')
shandler = logging.StreamHandler()
fhandler.setFormatter(FORMAT)
shandler.setFormatter(FORMAT)
log.addHandler(fhandler)
log.addHandler(shandler)

log.info('Starting EDA Script')

class mongo_connection():
    '''Creates an instance of a connection to the mongo DB'''
    def __init__(self):
        self._uri = 'mongodb://app:3knvak3ijs@localhost/kiva'
        try:
            self.client = mongodb_proxy.MongoProxy(pymongo.MongoClient(self._uri))        
            self.db = self.client['kiva']
        except:
            log.error('Could not establish a connection to Mongo Client')
            
def plt_distribution(var):
    black_list = ['use', 'activity']
    if var in black_list or 'description' in var:
        return  # Don't try to plot text variables

    try:
        lower_bound = int(math.floor(min([del_loans[var].min(), def_loans[var].min()])))
        upper_bound = int(math.ceil(max([del_loans[var].max(), def_loans[var].max()])))
        binwidth = int(math.ceil((upper_bound - lower_bound)/20))
        binwidth = 1 if binwidth == 0 else binwidth
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1,0.75, 0.75])
        if del_loans[var].dtype.name == 'float64' or del_loans[var].dtype.name == 'int64':
            fig = del_loans[var].hist(alpha=.5,
                                      color='green',
                                      bins=xrange(lower_bound, upper_bound+binwidth, binwidth),
                                      weights=np.zeros_like(del_loans[var]) + 1. / del_loans[var].size,
                                      label='Repaid')
            if var != 'dollar_days_late_metric':
                def_loans[var].hist(alpha=.5,
                                    color='red',
                                    bins=xrange(lower_bound, upper_bound+binwidth, binwidth),
                                    weights=np.zeros_like(def_loans[var]) + 1. / def_loans[var].size,
                                    label='Defaulted')
        if del_loans[var].dtype.name == 'object':
            fig = del_loans[var].plot(kind='bar',
                                      alpha=.5,
                                      color='green',
                                      bins=xrange(lower_bound, upper_bound+binwidth, binwidth),
                                      weights=np.zeros_like(del_loans[var]) + 1. / del_loans[var].size,
                                      label='Repaid')
            if var != 'dollar_days_late_metric':
                def_loans[var].plot(kind='bar',
                                    alpha=.5,
                                    color='red',
                                    bins=xrange(lower_bound, upper_bound+binwidth, binwidth),
                                    weights=np.zeros_like(def_loans[var]) + 1. / def_loans[var].size,
                                    label='Defaulted')
        mu = np.average(del_loans[var])
        sigma = np.std(del_loans[var])
        textstr = 'Repaid\n$\mu=%.3f$\n$\sigma=%.3f$'%(mu, sigma) 
        props = dict(boxstyle='round', facecolor='#336600', alpha=0.5)
        ax.text(1.02, 0.95, textstr, fontsize=14, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
        plt.axvline(x=mu, color='#336600', linewidth=3.0)
        plt.axvline(x=mu - sigma, color='#336600', linewidth=1.0, alpha=.5)
        plt.axvline(x=mu + sigma, color='#336600', linewidth=1.0, alpha=.5)
        mu = np.average(def_loans[var])
        sigma = np.std(def_loans[var])

        ignore_default = ['dollar_days_late_metric', 'actual_days_to_pay']
        if var not in ignore_default:
            textstr = 'Defaulted\n$\mu=%.3f$\n$\sigma=%.3f$'%(mu, sigma)
            props = dict(boxstyle='round', facecolor='#990000', alpha=0.5)
            ax.text(1.02, 0.72, textstr, fontsize=14, transform=ax.transAxes,
                    verticalalignment='top', bbox=props)
            plt.axvline(x=mu, color='#990000', linewidth=3.0)
            plt.axvline(x=mu-sigma, color='#990000', linewidth=1.0, alpha=.5)
            plt.axvline(x=mu+sigma, color='#990000', linewidth=1.0, alpha=.5)

            #One Way ANOVA Between Defaulted and Repaid
            f_val, p_val = f_oneway(del_loans[var], def_loans[var])
            textstr = 'ANOVA\np=%.3f'%(p_val)
            props = dict(boxstyle='round', facecolor='white')
            ax.text(1.02, 0.5, textstr, fontsize=14, transform=ax.transAxes,
                     verticalalignment='top', bbox=props)

        plt.title('%s Distribution' % ' '.join([s.capitalize() for s in var.split('_')]))
        plt.grid(False)
        path = './figs/distributions/%s.png' % var
        fig.get_figure().savefig(path) 
    except Exception as e:
        log.error('Could not make a dist plot for %(var)s because of %(e)s' % {'var': var, 'e': e})

if __name__ == '__main__':
    mongo_conn = mongo_connection()
    kiva_db = mongo_conn.db     
    flat_loan_collection = kiva_db['flat_loans']

    #Make directories for saving images
    directories = ['./figs','./figs/distributions',
                   './figs/y_scatter',
                   './figs/y_scatter/defaulted',
                   './figs/y_scatter/delinquent']
    for folder in directories:
        if os.path.isdir(str(os.sep).join([os.getcwd(), folder])) is False:
            os.mkdir(str(os.sep).join([os.getcwd(), folder]))
        
    obs_count = flat_loan_collection.find().count()
    def_count = flat_loan_collection.find({'defaulted':1}).count()
    del_count = obs_count - def_count

    log.debug('Of the %(obs)i, we had %(def)i default percentage and %(del)i delinquency' %
              {'obs': obs_count, 'def': def_count, 'del': del_count})

    cursor = flat_loan_collection.find()
    loans = list(cursor)
    
    loans = pd.DataFrame(loans)
    loans.fillna(value=0,inplace=True)

    #Remove variables that are populated during the life of the loan or are unhelpful
    remove_vars = ['translator_byline',
                   'translator_image',
                   'video_title',
                   'video_id',
                   'image_template_id',
                   'image_id',
                   'video_thumbnailImageId',
                   'video_youtubeId']
    [loans.drop(var, 1, inplace=True) for var in remove_vars]
    log.info(loans.describe())    
    
    del_loans = loans.ix[loans['defaulted'] == 0]
    def_loans = loans.ix[loans['defaulted'] == 1]

    pool = mp.Pool()
    pool.map(plt_distribution, loans.keys())
    pool.close()
    pool.join()
    pool.terminate()

    #Arrears distribution by the life of the loan
    fig = plt.figure()
    del_deciles = ['delinquency_decile_%s' % i for i in xrange(1, 11)]
    del_loans.boxplot(column=del_deciles)
    plt.title('Arrears Distribution by Life of Loan')
    plt.xlabel('Life of Loan')
    plt.ylabel('Pct. of Loan Value in Arrears')
    plt.xticks(xrange(1, 11), [' '.join([str(i), '%']) for i in xrange(10, 101, 10)])
    plt.ylim(-1, 1.05)
    fig.savefig('./figs/del_deciles.png')

    #Find out which variables have a significant correlation with dollar days late. Make scatter plots
    keys = list()

    black_list = ['_id',
                  'activity',
                  'use']

    for k in del_loans.keys():
        if k not in black_list and 'description' not in k:
            try:
                keys.append({'var_name': k, 'pearson': pearsonr(del_loans['dollar_days_late_metric'], del_loans[k])})
            except:
                log.debug('Could not calculate Pearson R for dollar days late and %s' %k)
    
    sig_keys = [k for k in keys if k['pearson'][1] < 0.05]
    for var in sig_keys:
        var = var['var_name']
        try:
            plt.figure()
            fig = plt.scatter(del_loans[var], del_loans['dollar_days_late_metric'], alpha=.2)
            plt.xlabel(' '.join([s.capitalize() for s in var.split('_')]))
            plt.ylabel('Dollar Days Late')
            plt.title('%s Delinquency Scatter Plot' % ' '.join([s.capitalize() for s in var.split('_')]))
            path = './figs/y_scatter/delinquent/ddl_scatter_%s.png' % var
            fig.get_figure().savefig(path) 
        except:
            log.error('Could not make a scatter plot with %s' % var)
       
    #Analyze Defaulted Loans with scatter plots as well
    keys = list()
    for k in loans.keys():
        if k not in black_list and 'description' not in k:
            try:
                keys.append({'var_name': k, 'pearson': pearsonr(loans['defaulted'], loans[k])})
            except:
                log.debug('Could not calculate Pearson R for defaulted and %s' % k)
            
    sig_keys = [k for k in keys if k['pearson'][1] < 0.05]
    
    for var in sig_keys:
        var = var['var_name']
        try:
            plt.figure()
            fig = plt.scatter(loans[var], loans['defaulted'], alpha=.2)
            plt.xlabel(' '.join([s.capitalize() for s in var.split('_')]))
            plt.ylabel('Defaulted')
            plt.title('%s Defaulted Scatter Plot' % ' '.join([s.capitalize() for s in var.split('_')]))
            path = './figs/y_scatter/defaulted/def_scatter_%s.png' % var
            fig.get_figure().savefig(path) 
        except:
            log.error('Could not make a scatter plot with %s' % var)