''' Dummy configuration file: Copy to ir-2016/config.py and set appropriate values '''

import os

# change to ':memory:' to create the database in memory
BASE_DATA_PATH = '/media/moloch/HHD/MachineLearning/data/yahoo_qa/'

DB_PATH = os.path.join(BASE_DATA_PATH, 'db.sqlite3')

DATASETS = {
    'small_sample': os.path.join(BASE_DATA_PATH, 'small_sample.xml'),
    'full_part_1': os.path.join(BASE_DATA_PATH, 'FullOct2007.xml.part1'),
    'full_part_2': os.path.join(BASE_DATA_PATH, 'FullOct2007.xml.part2')
}

# maximum string lengths, mostly for specifying the database model
STRING_LENGTHS = {
    'question_title': 200,
    'question_content': 2000,
    'answer_content': 2000,
    'category': 100,
}
