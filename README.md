

# Predict Kiva

Use Kiva's data snapshot (http://www.build.kiva.org) to predict which loans' delinquency and default. This code was created as part of Matthew Crowson's thesis requirement for graduation from 
Northwestern University's Master of Science in Predictive Analytics. The code is released to allow Kiva and others in microfinance to help lenders and lending
partners understand the risk involved with specific loans.

## Install
You'll need a few things to run the script. This was built using Python 2.7 and a local instance of Mongo (get verion).

* Download the JSON data snapshot from kiva: http://s3.kiva.org/snapshots/kiva_ds_json.zip
* pip install the following: pandas, scikit-learn, pymongo
* Checkout the code: 'git clone git://github.com/mcrowson/predict-kiva.git' 

## Setup
There are four major activities the script takes: load the JSON documents into Mongo, flatten the mongo documents and create some new variables, exploratory data analysis, and create the predictive models.

* **Load Documents Into Mongo** - filename.py: describe the file
* **Flatten and Prepare** - filename.py: describe the file
* **Exploratory Data Analysis** - filename.py: describe the file
* **Modeling** - filename.py: describe the file
 
Depending upon your computer's computing power and RAM, these scripts could take between an hour and a day to run completely. Multiprocessing is used where allowed by Python and the default settings are to use all available cores.

## Final Models
The best trained models are pickled and saved in this repo as well. These are scikit-learn objects of RandomForestRegressor and RandomForestClassifier. To predict new kiva loans, new observations must be flattened and prepared prior to using these objects predict method. 
