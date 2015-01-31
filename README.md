

# Predict Kiva

Use Kiva's data snapshot (http://www.build.kiva.org) to predict which loans' delinquency and default. This code was created as part of Matthew Crowson's thesis requirement for graduation from 
Northwestern University's Master of Science in Predictive Analytics. The code is released to allow Kiva and others in microfinance to help lenders and lending
partners understand the risk involved with specific loans.

## Install
You'll need a few things to run the script. This was built using Python 2.7 and a local instance of Mongo 2.6.3.

* Download the JSON data snapshot from kiva: http://s3.kiva.org/snapshots/kiva_ds_json.zip
* Checkout the code: 'git clone git://github.com/mcrowson/predict-kiva.git' 
* cd predict-kiva && pip install -r requirements.txt


## Setup
There are four major activities the script takes: load the JSON documents into Mongo, flatten the mongo documents and create some new variables, exploratory data analysis, and create the predictive models.

* **Load Documents Into Mongo** - jsontomongo.py: Takes the JSON snapshot file and loads it into your mongo database as-is.
* **Flatten and Prepare** - flatten_and_prepare.py: Takes the multi-level loan dictionaries and falttens them into a single table for predictive modeling. Variables are added, changed, and removed where applicable.
* **Exploratory Data Analysis** - EDA.py: Creates distribution plots for each variable, correlation plots for delinquency and default for each variable, and the dollar days late boxplots over the life of the loan. 
* **Modeling** - modeling.py: Creates the delinquency and default predictive models.

Depending upon your computer's computing power and RAM, these scripts could take between an hour and a day to run completely. Multiprocessing is used where allowed by Python and the default settings are to use all available cores.

## Final Models
It was my intention to save the final models on this repo. However, their size (severl hundred meg for the random forest models) prevented their inclusion. The modeling script saves the final models to your local directory as pickled objects. These are scikit-learn objects of RandomForestRegressor and LogisticRegressionClassifier. To predict new kiva loans, new observations must be flattened and prepared prior to using these objects predict method. The files can then be unpickled as trained objects. These trained objects can then score the new observations.
