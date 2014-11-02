# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 08:17:35 2013


I could just see how many other loans are near them for determining geo/urban

I could see how many other lending partners made loans near them to approximate
access to capital



Kiva Snapshot retrieved on 6/23/14



To do:
Match bar and histogram distributions.
Have a mean and standard deviation for each
For boolean variables, have a split between default and delinquency.

@author: Matthew
"""

import pymongo
from utils import mongodb_proxy
import logging
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr, f_oneway
import random as ran
import multiprocessing as mp
import numpy as np
import os
import math

#Script Logging
LEVEL = logging.DEBUG
FORMAT = logging.Formatter('%(asctime)-15s %(name)s %(levelname)-8s %(message)s')
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
    try:
        lower_bound = int(math.floor(min([del_loans[var].min(),def_loans[var].min()])))
        upper_bound = int(math.ceil(max([del_loans[var].max(),def_loans[var].max()])))
        binwidth = int(math.ceil((upper_bound - lower_bound)/20))
        binwidth = 1 if binwidth==0 else binwidth
        fig = plt.figure()
        ax =  fig.add_axes([0.1, 0.1,0.75, 0.75])
        if del_loans[var].dtype.name == 'float64' or del_loans[var].dtype.name == 'int64':
            fig = del_loans[var].hist(alpha=.5, color='green', bins=xrange(lower_bound,upper_bound+binwidth,binwidth), weights=np.zeros_like(del_loans[var]) + 1. / del_loans[var].size, label='Repaid')
            if var != 'dollar_days_late':
                def_loans[var].hist(alpha=.5, color='red', bins=xrange(lower_bound,upper_bound+binwidth,binwidth), weights=np.zeros_like(def_loans[var]) + 1. / def_loans[var].size, label='Defaulted')
        if del_loans[var].dtype.name == 'object':
            fig = del_loans[var].plot(kind='bar', alpha=.5, color='green', bins=xrange(lower_bound,upper_bound+binwidth,binwidth), weights=np.zeros_like(del_loans[var]) + 1. / del_loans[var].size, label='Repaid')
            def_loans[var].plot(kind='bar', alpha=.5, color='red', bins=xrange(lower_bound,upper_bound+binwidth,binwidth), weights=np.zeros_like(def_loans[var]) + 1. / def_loans[var].size, label='Defaulted')
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
        textstr = 'Defaulted\n$\mu=%.3f$\n$\sigma=%.3f$'%(mu, sigma)  
        props = dict(boxstyle='round', facecolor='#990000', alpha=0.5)
        ax.text(1.02, 0.72, textstr, fontsize=14, transform=ax.transAxes,
                 verticalalignment='top', bbox=props)        
        plt.axvline(x=mu, color = '#990000', linewidth=3.0)
        plt.axvline(x=mu-sigma,color='#990000', linewidth=1.0, alpha=.5)
        plt.axvline(x=mu+sigma,color='#990000', linewidth=1.0, alpha=.5)
        #One Way ANOVA Between Defaulted and Repaid
        f_val, p_val = f_oneway(del_loans[var],def_loans[var])
        textstr = 'ANOVA\np=%.3f'%(p_val)  
        props = dict(boxstyle='round', facecolor='white')
        ax.text(1.02, 0.5, textstr, fontsize=14, transform=ax.transAxes,
                 verticalalignment='top', bbox=props)    
        plt.title('%s Distribution' % ' '.join([s.capitalize() for s in var.split('_')]))
        plt.grid(False)
        path = 'figs/distributions/%s.png' % var
        fig.get_figure().savefig(path) 
    except Exception as e:
        log.error('Could not make a dist plot for %(var)s because of %(e)s' % {'var':var,'e':e})

if __name__ == '__main__':
    mongo_conn = mongo_connection()
    kiva_db = mongo_conn.db     
    flat_loan_collection = kiva_db['flat_loans']

    #Make directories for saving images
    directories = ['figs','figs/distributions','figs/y_scatter', 'figs/y_scatter/defaulted','figs/y_scatter/delinquent']
    for folder in directories:
        if  os.path.isdir(str(os.sep).join([os.getcwd(),folder])) == False:
            os.mkdir(str(os.sep).join([os.getcwd(),folder]))
    

    #Get Samples for faster graphing. 
        
    obs_count = flat_loan_collection.find().count()
    def_count = flat_loan_collection.find({'defaulted':1}).count()
    del_count = obs_count - def_count
    '''    
    t_score = 2.575
    margin_of_error = .01
    sigma_ddl = np.std([l['dollar_days_late_metric'] for l in flat_loan_collection.find({'defaulted':0},{'dollar_days_late_metric':1})])
    sigma_def = np.std()
    ss = (pow(t_score,2)*def_prop*del_prop)/pow(margin_of_error,2)
    sample_size = ss / (1 + ((ss - 1)/obs_count))#Adjust for Finite Population
    '''    
    sample_size = (0.05 * obs_count)
    log.debug('Of the %(obs)i, we had %(def)i default percentage and %(del)i delinquency' % {'obs':obs_count,'def':def_count,'del':del_count})

    

    ran.seed(8798)
    threshold = ran.random()
    log.debug('Random value is %s' % threshold)
    #c = 0
    #while c < sample_size: #Go until we have enough observations for the sample size
    #    cursor = flat_loan_collection.find({'random': {'$gte': threshold}})
    #    c = cursor.count()
    cursor = flat_loan_collection.find()
    loans = [l for l in cursor[:int(sample_size)]]
    log.debug('The sample is %(size)i large with %(def)i defaulted loans' % {'size':len(loans),'def':sum([1 for l in loans if l['defaulted']==1])})    
    
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
    [loans.drop(var,1,inplace=True) for var in remove_vars]
    log.info(loans.describe())    
    
    del_loans = loans.ix[loans['defaulted'] == 0]
    def_loans = loans.ix[loans['defaulted'] == 1]

    pool = mp.Pool()
    pool.map(plt_distribution,loans.keys())  
    pool.close()
    pool.join()
    pool.terminate()

    #Arrears distribution by the lift of the loan
    fig = plt.figure()
    del_deciles = ['delinquency_decile_%s' % i for i in xrange(1,11)]
    del_loans.boxplot(column=del_deciles)
    plt.title('Arrears Distribution by Life of Loan')
    plt.xlabel('Life of Loan')
    plt.ylabel('Pct. of Loan Value in Arrears')
    plt.xticks(xrange(2,11,2),[' '.join([str(i/10),'%']) for i in xrange(20,101,20)])
    plt.ylim(-1.05, 1.05)
    fig.savefig('figs/del_deciles.png')    

    #Find out which variables have a significant correlation with dollar days late. Make scatterplots
    keys = list()    
    for k in del_loans.keys():
        try:
            keys.append({'var_name':k,'pearson':pearsonr(del_loans['dollar_days_late_metric'],del_loans[k])})
        except:
            log.debug('Could not calculate Pearson R for dollar days late and %s' %k)
    
    sig_keys = [k for k in keys if k['pearson'][1] < 0.05]
    for var in sig_keys:
        var = var['var_name']
        try:
            plt.figure()
            fig = plt.scatter(del_loans[var],del_loans['dollar_days_late_metric'], alpha=.2) 
            plt.xlabel(var)
            plt.ylabel('dollar_days_late_metric')
            path = 'figs/y_scatter/delinquent/ddl_scatter_%s.png' % var
            fig.get_figure().savefig(path) 
        except:
            log.error('Could not make a scatter plot with %s' % var)
       
    #Analyze Defaulted Loans with scatterplots as well
    keys = list()
    for k in loans.keys():
        try:
            keys.append({'var_name':k,'pearson':pearsonr(loans['defaulted'],loans[k])})
        except:
            log.debug('Could not calculate Pearson R for defaulted and %s' %k)
            
    sig_keys = [k for k in keys if k['pearson'][1] < 0.05]
    
    for var in sig_keys:
        var = var['var_name']
        try:
            plt.figure()
            fig = plt.scatter(loans[var],loans['defaulted'], alpha=.2)   
            plt.xlabel(var)
            plt.ylabel('Defaulted')
            path = 'figs/y_scatter/defaulted/def_scatter_%s.png' % var
            fig.get_figure().savefig(path) 
        except:
            log.error('Could not make a scatter plot with %s' % var)