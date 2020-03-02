from sqlalchemy.sql import exists
from sqlalchemy import exc, desc
from Review.app import db


class Gene:
    number = db.Column(db.Integer, primary_key=True, nullable=False)
    description = db.Column(db.String, nullable=False)
