""" Defines the database model for questions and answers (also categories, if they are present) """

import os
import string

import config

from sqlalchemy import Column, String, Boolean, Date, create_engine, ForeignKey, Integer
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# ---------------
# DATABASE MODELS
# ---------------


def truncate(text, length=10):
    if text is None:
        return ''

    text_clean = filter(lambda x: x in set(string.printable), text)
    return text_clean[:length] + '...' if len(text_clean) > length else text_clean


class Category(Base):
    __tablename__ = 'category'

    id = Column(Integer, primary_key=True)
    text = Column(String(config.STRING_LENGTHS['category']))

    def __repr__(self):
        return u'<Category: id=%d, text="%s">' % (self.id, self.text)


class Question(Base):
    __tablename__ = 'question'

    id = Column(Integer, primary_key=True)

    # main information
    title = Column(String(config.STRING_LENGTHS['question_title']))
    content = Column(String(config.STRING_LENGTHS['question_content']))

    # foreign key to category (instead of storing category id in each one)
    category_id = Column(Integer, ForeignKey('category.id'))
    category = relationship(Category)

    # metadata
    date = Column(Date, unique=False, nullable=True)
    res_date = Column(Date, unique=False, nullable=True)
    vot_date = Column(Date, unique=False, nullable=True)
    yahoo_id = Column(String(20), nullable=True)
    best_answer_yahoo_id = Column(String(20), nullable=True)

    def __repr__(self):
        return u'<Question: id=%d, title="%s", content="%s", category="%s">' % (self.id,
                                                                                truncate(self.title),
                                                                                truncate(self.content),
                                                                                truncate(self.category.text))


class Answer(Base):
    __tablename__ = 'answer'

    id = Column(Integer, primary_key=True)

    # contents of the question and whether or not it was the "best answer"
    content = Column(String(config.STRING_LENGTHS['answer_content']))
    is_best = Column(Boolean, unique=False, default=False)

    # foreign key to question
    question_id = Column(Integer, ForeignKey('question.id'))
    question = relationship(Question)

    def __repr__(self):
        return '<Answer: id=%d, title="%s", is_best=%r>' % (self.id, truncate(self.content), self.is_best)

_engine = create_engine('sqlite:///' + config.YAHOO_DB_PATH)
Base.metadata.bind = _engine
DBSession = sessionmaker(bind=_engine)


def init_db(test=False, test_num=10):
    if os.path.isfile(config.YAHOO_DB_PATH):
        print('Removing "%s"...' % config.YAHOO_DB_PATH)
        os.remove(config.YAHOO_DB_PATH)

    print('Creating database at "%s"...' % config.YAHOO_DB_PATH)
    Base.metadata.create_all(_engine)

    def test_db(num):
        """ Run after creating a new database to ensure that it works as anticipated. """

        print('\n*** database unit test ***')

        session = DBSession()

        categories = [Category(text='dummy category %d' % i) for i in range(num)]
        questions = [Question(title='dummy question %d' % i, content='this is a dummy question', category=categories[i]) for i in range(num)]
        answers = [Answer(content='dummy answer %d' % i, question=questions[i]) for i in range(num)]
        session.add_all(categories + questions + answers)
        session.commit()

        print('Added %d dummy categories, questions and answers' % num)

        categories = session.query(Category).all()
        assert len(categories) == num
        print('Categories: {}'.format(categories))

        questions = session.query(Question).all()
        assert len(questions) == num
        print('Questions: {}'.format(questions))

        answers = session.query(Answer).all()
        assert len(answers) == num
        print('Answers: {}'.format(answers))

        for i in range(3):
            print('Answers to Question {}, {}: {}'.format(i,
                                                          questions[i],
                                                          session.query(Answer).filter(Answer.question == questions[i]).all()))

        for e in categories + questions + answers:
            session.delete(e)
        print('Deleted all dummy categories, questions and answers')

        assert session.query(Category).count() == 0
        assert session.query(Question).count() == 0
        assert session.query(Answer).count() == 0
        print('Categories: {}, Questions: {}, Answers: {}'.format(session.query(Category).all(),
                                                                  session.query(Question).all(),
                                                                  session.query(Answer).all()))

        print('*** end of unit test ***\n')
        session.commit(); session.close()

    # comment out to remove testing
    if test:
        test_db(test_num)


if __name__ == '__main__':
    init_db(test=True)
