import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///' + os.path.join(basedir, 'database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = b'9\x93qa\xdd\xc8\xd1\x1a\xb1r\x83_\x9e\xb9\x02\xbe'
    COUNTER_CONFIG = "config/counter_config.json"
    COMBINER_CONFIG = "config/combiner_config.json"


class TestConfig(object):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL_TEST') or 'sqlite:///' + os.path.join(basedir, 'app_test.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a-very-secret-key'
