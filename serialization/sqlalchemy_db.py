import os
import sys

from sqlalchemy.orm import relationship, sessionmaker

import config

from sqlalchemy import Column, String, Boolean, Date, create_engine, ForeignKey, Integer
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# ---------------
# DATABASE MODELS
# ---------------


def truncate(text, length=10):
    return text[:length] + '...' if len(text) > length else text


class Category(Base):
    __tablename__ = 'questions'

    id = Column(Integer, primary_key=True)
    text = Column(String(config.STRING_LENGTHS['category']))

    def __repr__(self):
        return '<Category: id=%d, text="%s">' % (self.id, truncate(self.text))


class Question(Base):
    __tablename__ = 'question'

    id = Column(Integer, primary_key=True)

    # main information
    title = Column(String(config.STRING_LENGTHS['question_title']))
    content = Column(String(config.STRING_LENGTHS['question_content']))
    # category_id = relationship(Integer, ForeignKey('category.id'))
    # category = relationship(Category)

    # metadata
    date = Column(Date, unique=False, nullable=True)
    res_date = Column(Date, unique=False, nullable=True)
    vot_date = Column(Date, unique=False, nullable=True)
    yahoo_id = Column(String(20), nullable=True)
    best_answer_yahoo_id = Column(String(20), nullable=True)

    def __repr__(self):
        return '<Question: id=%d, title="%s", content="%s">' % (self.id, truncate(self.title), truncate(self.content))


class Answer(Base):
    __tablename__ = 'answer'

    id = Column(Integer, primary_key=True)

    # contents of the question and whether or not it was the "best answer"
    content = Column(String(config.STRING_LENGTHS['answer_content']))
    is_best = Column(Boolean, unique=False, default=False)

    # question_id = Column(Integer, ForeignKey('question.id'))
    # question = relationship(Question)

    def __repr__(self):
        return '<Answer: id=%d, title="%s", is_best=%r>' % (self.id, truncate(self.content), self.is_best)

assert os.path.isfile(config.DB_PATH), 'Database not found at "%s", run serialization/sqlalchemy_db.py' % config.DB_PATH
_engine = create_engine('sqlite:///' + config.DB_PATH)
Base.metadata.bind = _engine
DBSession = sessionmaker(bind=_engine)

if __name__ == '__main__':

    if os.path.isfile(config.DB_PATH):
        print('Removing "%s"...' % config.DB_PATH)
        os.remove(config.DB_PATH)

    print('Creating database at "%s"...' % config.DB_PATH)
    Base.metadata.create_all(_engine)

    print('Adding dummy question...')
    session = DBSession()

    session.add(Category(text='dummy category 1'))
    session.add(Category(text='dummy category 2'))
    session.commit()

    categories = session.query(Category).all()
    print(categories)

