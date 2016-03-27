from __future__ import print_function

import os
import sys
import config

import xml.etree.ElementTree as ET

from serialization.sqlalchemy_db import DBSession

dataset = 'small_sample'

# make sure the file exists so it can be processed
if not os.path.exists(config.DATASETS[dataset]):
    print('File not found at: "%s"' % config.DATASETS[dataset])
    sys.exit(-1)

session = DBSession()

for event, elem in ET.iterparse(config.DATASETS[dataset], events=('start', 'end', 'start-ns', 'end-ns')):
    if event == 'end' and elem.tag == 'answer_item':
        print(elem.tag)