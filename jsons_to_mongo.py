# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:11:54 2014

@author: Matthew Crowson

The script takes an unzipped JSON snapshot from Kiva and inserts it
into a Mongo database named kiva with collection names of lenders, loans
and loans_lenders in accordance with the JSON folders from Kiva

Kiva Snapshot Retrieved on 6/23/14
"""


#Maybe organize these imports later by their usage?
import mongodb_proxy
import pymongo
import simplejson as json
import os
import logging

#Script Logging
LEVEL = logging.DEBUG
FORMAT = logging.Formatter('%(asctime)-15s %(name)s %(levelname)-8s %(message)s')
log = logging.getLogger(__name__)
log.setLevel(LEVEL)
fhandler = logging.FileHandler('jsons_to_mongo.log')
shandler = logging.StreamHandler()
shandler.setLevel(logging.ERROR)
fhandler.setFormatter(FORMAT)
shandler.setFormatter(FORMAT)
log.addHandler(fhandler)
log.addHandler(shandler)

log.debug('Starting Kiva Script')

class mongo_connection():
    '''Creates an instance of a connection to the mongo DB'''
    def __init__(self):
        self._uri = 'mongodb://app:3knvak3ijs@localhost/kiva'
        try:
            self.client = mongodb_proxy.MongoProxy(pymongo.MongoClient(self._uri))        
            self.db = self.client['kiva']
        except:
            log.error('Could not establish a connection to Mongo Client')

def json_to_mongo(json_file, collection):
    '''Takes a JSON file and writes it to the appropriate Mongo collection'''    
    mongo_col = kiva_db[collection]    
    with open(json_file,'r') as json_data:     
        with_header = json.load(json_data)
        list_of_dicts = with_header[collection] #the subdoc with the name of the colleciton
    for d in list_of_dicts:
        try:
            mongo_col.insert(d)
        except:
            log.info('Could not insert the dict. Unique ids enforced.')
    return
    
    
def process_kiva_snapshot_files(containing_folder):
    '''Finds all of the json files from the kiva snapshot and
    writes them to the mongo db'''
    
    collections = {'lenders':[],'loans':[],'loans_lenders':[]}
    
    #populate file names    
    for col in collections.keys():
        path = os.path.join(containing_folder,col)
        log.debug('Getting files from %s' % path)
        try: #Try to get folder names matching our collections
            file_names = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
            collections[col] += file_names
            log.debug('Got %(count)i JSON files in the %(collection)s collection' % {'count':len(file_names),'collection':col})
        except:
            log.error('Could not get files at location. Folders/files may not exist')
    
    #write files to mongo
    for col, files in collections.iteritems():        
       # try:
        [json_to_mongo(f, col) for f in files]
        #except:
         #   log.error('Could not write the %s jsons to mongo' % col )

if __name__ == '__main__':
    mongo_conn = mongo_connection()
    kiva_db = mongo_conn.db
    #Where did you unzip the JSON zip file?
    containing_folder = '/Users/matthew/Downloads/kiva_ds_json'
    process_kiva_snapshot_files(containing_folder)
    
    #Create indexes for the checked flag and Kiva's id
    kiva_db['loans'].ensure_index({'id':1})
   
    log.handlers = [] #Remove old handlers so we don't have a bunch of 'em