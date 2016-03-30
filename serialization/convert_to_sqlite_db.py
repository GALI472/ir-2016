""" Intended to be run as a stand-alone script (not included in __init__.py) """

from __future__ import print_function

import os
import sys

import datetime
import config
import xml.etree.ElementTree as ET

from serialization.sqldb import DBSession, Category, Question, Answer, init_db

import logging
logger = logging.getLogger(__name__)

DO_YOU_REALLY_WANT_TO_RUN = False
assert DO_YOU_REALLY_WANT_TO_RUN, 'Do you REALLY want to run? The database takes a long time to create!'

dataset = 'yahoo_full'

# make sure the file exists so it can be processed
if not os.path.exists(config.DATASETS[dataset]):
    logger.info('File not found at: "%s"' % config.DATASETS[dataset])
    sys.exit(-1)

session = DBSession()

# initialize the database
init_db(config.DATABASES['yahoo'], test=True)

# smallest date: dataset provides days relative to this date
first_day = datetime.date(day=1, month=1, year=1970)

# we want to avoid adding an answer if the question isn't also added
question = Question()
answers = []

count = 0

for event, elem in ET.iterparse(config.DATASETS[dataset], events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'end':
        if elem.tag == 'document':
            session.add_all([question] + answers)

            count += 1
            if count % 10000 == 0:
                logger.info('Processed %d questions' % count)
                session.commit()

            # clear variables being stored
            question = Question()
            answers = []

        elif elem.tag == 'subject':
            question.title = elem.text.strip()

        elif elem.tag == 'content':
            question.content = elem.text.strip()

        elif elem.tag == 'bestanswer' or elem.tag == 'answer_item':
            answer = Answer(content=elem.text, is_best=elem.tag == 'bestanswer', question=question)
            answers.append(answer)

        elif elem.tag == 'cat':
            categories = session.query(Category).filter(Category.text==elem.text.strip())
            if categories.count() == 0:
                category = Category(text=elem.text.strip())
                question.category = category
            else:
                question.category = categories.first()

        elif elem.tag == 'date':
            question.date = first_day + datetime.timedelta(seconds=int(elem.text))

        elif elem.tag == 'res_date':
            question.res_date = first_day + datetime.timedelta(seconds=int(elem.text))

        elif elem.tag == 'vot_date':
            question.res_date = first_day + datetime.timedelta(seconds=int(elem.text))

        elif elem.tag == 'id':
            question.yahoo_id = elem.text.strip()

        elif elem.tag == 'best_id':
            question.best_answer_yahoo_id = elem.text.strip()

# commit to database and close the session
logger.info('Done processing data; committing extra changes to database and closing session')
session.commit(); session.close()
