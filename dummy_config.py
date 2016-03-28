""" Dummy configuration file: Copy to ir-2016/config.py and set appropriate values """

import os

# change to ':memory:' to create the database in memory
BASE_DATA_PATH = '/media/moloch/HHD/MachineLearning/data/ir-2016/'

YAHOO_DB_PATH = os.path.join(BASE_DATA_PATH, 'yahoo.sqlite3')

DATASETS = {
    'yahoo_small_sample': os.path.join(BASE_DATA_PATH, '../yahoo_qa/small_sample.xml'),
    'yahoo_full': os.path.join(BASE_DATA_PATH, '../yahoo_qa/FullOct2007.xml.total'),
}

# maximum string lengths, mostly for specifying the database model
STRING_LENGTHS = {
    'question_title': 140,
    'question_content': 1500,
    'answer_content': 10000,
    'category': 100,
}

