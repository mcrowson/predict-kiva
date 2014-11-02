# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 08:17:35 2013


I could just see how many other loans are near them for determining geo/urban

I could see how many other lending partners made loans near them to approximate
access to capital

mongo admin creds:
admin: root

Kiva Snapshot retrieved on 6/23/14


@author: Matthew
"""

import pymongo
from utils import mongodb_proxy
import collections 
import multiprocessing
from datetime import datetime, timedelta
import logging
import random as ran
import sys

#Script Logging
LEVEL = logging.INFO
FORMAT = logging.Formatter('%(asctime)-15s %(name)s %(levelname)-8s %(message)s')
log = logging.getLogger(__name__)
log.setLevel(LEVEL)
fhandler = logging.FileHandler('flatten_and_prepare.log')
shandler = logging.StreamHandler()
fhandler.setFormatter(FORMAT)
shandler.setFormatter(FORMAT)
log.addHandler(fhandler)
log.addHandler(shandler)

log.info('Starting Flatten and Prepare Script')


class MongoConnection():
    '''
    Creates an instance of a connection to the mongo DB
    '''
    def __init__(self):
        self._uri = 'mongodb://app:3knvak3ijs@localhost/kiva'
        try:
            self.client = mongodb_proxy.MongoProxy(pymongo.MongoClient(self._uri))        
            self.db = self.client['kiva']
        except:
            log.error('Could not establish a connection to Mongo Client')


def chunk_me(length, n):
    """ 
        Takes a range of length , and turns it into a list of n items whose values
        total l.
    """
    chunks = []
    for i in xrange(0, len(length), n):
        chunks.append(sum(length[i:i+n]))
    return chunks

def flatten(d, parent_key=''):
    '''Takes a multi-level document and returns a flat dictionary'''
    items = []
    for k, v in d.items():
        new_key = parent_key + '_' + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)
    
    
def date_range(start_date, end_date):
    '''Date Generator'''
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n) 
    
    
def process_loan(loan_counter):
    '''
    Processes a single loan
    '''
    loan = loan_collection.find_one({'id': loan_counter})
    #Create new variabels
 
    
    #Date Variable Creation
    #Turn date string into date obj : = datetime.strptime(string_var,date_format)
    date_vars = ['posted_date', 'planned_expiration_date', 'funded_date', 'paid_date']
    
    for var in date_vars:
        if type(loan[var]) == unicode:
            loan[var] = datetime.strptime(loan[var], date_format)
            
    for pmt in loan['payments']:  # Payment dates to date obj
        if type(pmt['settlement_date']) == unicode:
            pmt['settlement_date'] = datetime.strptime(pmt['settlement_date'], date_format)
        pmt['currency_exchange_loss_amount'] = float(pmt['currency_exchange_loss_amount'])

    for sched in loan['terms']['scheduled_payments']:  # Scheduled Payment dates to date obj
        if type(sched['due_date']) == unicode:
            sched['due_date'] = datetime.strptime(sched['due_date'], date_format)

    if type(loan['terms']['disbursal_date']) == unicode:
        loan['terms']['disbursal_date'] = datetime.strptime(loan['terms']['disbursal_date'], date_format)
    
    #Random Value for Sampling
    loan['random'] = ran.random()    
    
    #Days to Raise Money
    loan['days_to_fund'] = (loan['funded_date'] - loan['posted_date']).days

    #Days to Pay Money
    loan['scheduled_days_to_pay'] = \
        (loan['terms']['scheduled_payments'][-1]['due_date'] - loan['terms']['disbursal_date']).days
   
    #Does a Video Exist
    loan['video_exists'] = 1 if loan['video'] is not None else 0
   
    #Borrower Count
    loan['borrower_count'] = len(loan['borrowers'])
    
    #Ensure the Exchange loss amount is an integer. yields 0 if Null value for exchange loss
    loan['currency_exchange_loss_amount'] = 0 if loan['currency_exchange_loss_amount'] is None \
        else loan['currency_exchange_loss_amount']
    
    #Bonus credit from boolean to int
    loan['bonus_credit_eligibility'] = 1 if loan['bonus_credit_eligibility'] is True else 0
    
    #% Borrower Female
    loan['pct_female'] = float(sum([1 for b in loan['borrowers'] if b['gender'] == 'F']))/float(loan['borrower_count'])
    
    #Geo pairs
    loan['location_geo_lat'], loan['location_geo_lon'] =\
        [float(x) for x in loan['location']['geo'].pop('pairs',' ').split(' ')]
           
    #Number of scheduled payments
    loan['scheduled_payment_count'] = len(loan['terms']['scheduled_payments'])
    
    #Days final payment was late
    last_sched_pmt = loan['terms']['scheduled_payments'][-1]['due_date']
    
    #Month of loan receipt
    loan['month_loan_received'] = loan['terms']['disbursal_date'].month
    
    #No payment to $0 payment
    if loan['paid_amount'] is None : loan['paid_amount'] = 0
    loan['loan_amount'] = float(loan['loan_amount'])
        
    #Calculate Dollar-Days Late / loan amount
    #If defaulted then it will be 'Defaulted' string, not sure how to handle
    if loan['paid_amount'] + loan['currency_exchange_loss_amount'] < loan['funded_amount']:
        loan['defaulted'] = 1
    else:  # Loan repaid in full, check for late payments
        loan['defaulted'] = 0
        loan['actual_days_to_pay'] = (loan['payments'][-1]['settlement_date'] - loan['terms']['disbursal_date']).days  
        last_act_pmt = loan['payments'][-1]['settlement_date']
        loan['final_pmt_days_late'] = (last_act_pmt -  last_sched_pmt).days
        dollar_days_late = float()
        arrears = float()
        last_day = max([last_sched_pmt, last_act_pmt])
        delinquency_decile = float()
        counter = int()
        
        for day in date_range(loan['terms']['disbursal_date'], (last_day + timedelta(days=1))):
            log.debug(arrears)
            log.debug(day)
            #See if scheduled payments are planned, add them to amount outstanding
            for sched in loan['terms']['scheduled_payments']:
                if sched['due_date'].date() == day.date():
                    arrears += float(sched['amount'])
                    log.debug(' '.join([str(sched['due_date']), 'arrears plus', str(sched['amount'])]))
        
            #See if payments were settled, subtract from amount outstanding
            for pmt in loan['payments']:
                if pmt['settlement_date'].date() == day.date() : 
                    arrears -= (pmt['amount'] + pmt['currency_exchange_loss_amount'])
                    log.debug(' '.join([str(pmt['settlement_date']), 'arrears minus', str(pmt['amount'])]))
        
            #Arrears at a decile of repayment

            pct_complete = float((day - loan['terms']['disbursal_date']).days) / loan['scheduled_days_to_pay']
            #log.debug('%(day_minus_disbursed)d divided by %(dayspay)d' % {'day_minus_disbursed':(day - loan['terms']['disbursal_date']).days,'dayspay':loan['scheduled_days_to_pay']})
            #log.debug('%s percent complete ' % pct_complete)
            if pct_complete >= float(delinquency_decile/10) and counter <= 10: #Don't track arrears after last scheduled pmt
                log.debug('%(pct)s is larger than %(dec)s' % {'pct': pct_complete,'dec': float(delinquency_decile/10)})
                name = '_'.join(['delinquency_decile', str(counter)])
                loan[name] = arrears/loan['loan_amount'] #Delinquency Rate
                log.debug('%(count)s: %(arrears)s divided by %(amt)s is %(final)s'
                          % {'count': counter, 'arrears': arrears, 'amt': loan['loan_amount'], 'final': loan[name]})
                log.debug('%(name)s has value of %(blank)s' % {'name':name,'blank':loan[name]})                
                delinquency_decile += 1 
                counter += 1
            
            #add amount outstanding to dollar days late
            dollar_days_late += arrears
    
        #Adjust for the amount of the loan
        loan['dollar_days_late_metric'] = dollar_days_late / loan['loan_amount']

    flat_loan = flatten(loan)    
    #Remove things we don't want to keep
    variables_to_remove = ["basket_amount",
                           "posted_date",
                           "planned_expiration_date",
                           "funded_date",
                           "paid_date",
                           "video",
                           "image",
                           "delinquent",
                           "status",
                           "translator",
                           "borrowers",
                           "payments",
                           'checked',
                           'description_languages',
                           'terms_scheduled_payments',
                           'terms_local_payments',
                           'terms_disbursal_date',
                           'name',
                           'location_town',
                           'location_country_code',
                           '_id']
     
    dummy_vars = ['sector',
                  'location_geo_level',
                  'terms_disbursal_currency',
                  'location_geo_type',
                  'terms_loss_liability_currency_exchange',
                  'terms_repayment_interval',
                  'terms_loss_liability_nonpayment',
                  'location_country',
                  'theme',
                  'partner_id']
    
    for var in dummy_vars:
        if flat_loan[var] is None:
            continue
        name = '_'.join([var, str(flat_loan[var])])
        flat_loan[name] = 1 

    [flat_loan.pop(var, None) for var in variables_to_remove + dummy_vars]
    try:
        flat_loan_collection.insert(flat_loan)
    except:
        log.error('Could not insert loan %s' % flat_loan['id'])


if __name__ == "__main__":
    date_format = '%Y-%m-%dT%H:%M:%SZ'

    mongo_conn = MongoConnection()
    kiva_db = mongo_conn.db     
    loan_collection = kiva_db['loans']
    
    kiva_db.create_collection('flat_loans')
    flat_loan_collection = kiva_db['flat_loans']
    
    #Pass an array of Kiva IDs to process.
    #Modify the find so it only gets loans I'm using in my research
    loans = [l['id'] for l in loan_collection.find({'status': {'$in': ['paid', 'defaulted']},
                                                    'terms.scheduled_payments.0': {'$exists': True},
                                                    'id': {'$lte': 725473}})]
                                                    
    log.info('Found %(count)s loans. Will now flatten and place in the new collection.' % {'count':len(loans)})
    num_procs = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=num_procs, maxtasksperchild=2)       
    pool.map(process_loan, loans)
    pool.close()
    pool.join()
    
    try:
        #Create index on Random values for faster sampling
        flat_loan_collection.ensure_index({'random': 1})
    except Exception as e:
        log.error('Could not create an index on random because of %s' % e)
    
    log.info('Flatten and Prepare Completed')

