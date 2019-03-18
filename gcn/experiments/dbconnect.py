# import mysql.connector
from sqlalchemy import create_engine
import pandas as pd

HOST = 'relational.fit.cvut.cz'
PORT = 3306
USER = 'guest'
PASSWORD = 'relational'
DB = 'CORA'

engine = create_engine(f'mysql+mysqlconnector://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB}')


feats_query = "SELECT " \
              "   paper.paper_id AS paper_id, " \
              "   paper.class_label as class_label, " \
              "   content.word_cited_id as word_cited_id " \
              "FROM paper " \
              "INNER JOIN content " \
              "   ON paper.paper_id = content.paper_id"
relats_query = "SELECT * " \
               "FROM cites"


feats = pd.read_sql(feats_query, engine)
relats = pd.read_sql(relats_query, engine)