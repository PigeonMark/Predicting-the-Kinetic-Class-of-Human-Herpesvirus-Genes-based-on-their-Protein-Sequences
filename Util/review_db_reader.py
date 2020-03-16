import json
import sqlite3
from sqlite3 import Error


class Review:
    def __init__(self, virus, names, old_phase, reviewed_phase, review_status):
        self.virus = virus
        self.names = names
        self.old_phase = old_phase
        self.reviewed_phase = reviewed_phase
        self.review_status = review_status


class ReviewDBReader:

    def __init__(self, config_filepath):
        self.connection = None  # type: sqlite3.Connection
        self.read_config(config_filepath)

    def read_config(self, config_filepath):
        with open(config_filepath) as config_file:
            config = json.load(config_file)

        try:
            self.connection = sqlite3.connect(config['db_file'])
        except Error as e:
            print(e)

    def get_all(self):
        cur = self.connection.cursor()
        cur.execute("SELECT * FROM gene")
        rows = cur.fetchall()
        ret_list = []
        for row in rows:
            ret_list.append(Review(row[0], row[1], row[2], row[3], row[4]))
        return ret_list

    def get_by(self, what, value):
        if what in ['virus', 'names', 'reviewed_phase', 'review_status']:
            ret_list = []
            rows = self.connection.cursor().execute(f"SELECT * FROM gene WHERE {what}=?", (value,)).fetchall()
            for row in rows:
                ret_list.append(Review(row[0], row[1], row[2], row[3], row[4]))
            return ret_list
        else:
            return []
