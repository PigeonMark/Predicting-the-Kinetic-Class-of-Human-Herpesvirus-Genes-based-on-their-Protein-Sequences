from flask import Flask
from Review.config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bootstrap import Bootstrap
from flask_fontawesome import FontAwesome
from flask_wtf.csrf import CSRFProtect
import os
import sys

db = SQLAlchemy()
migrate = Migrate()
bootstrap = Bootstrap()
fa = None
csrf = CSRFProtect()


def create_app(config_class=Config):
    if getattr(sys, 'frozen', False):
        template_folder = os.path.join(sys._MEIPASS, 'app/templates')
        static_folder = os.path.join(sys._MEIPASS, 'app/static')
        app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
    else:
        app = Flask(__name__)
    app.config.from_object(config_class)

    # from .models import Product, Table, Order, Payment

    db.init_app(app)
    migrate.init_app(app, db)
    bootstrap.init_app(app)
    global fa
    fa = FontAwesome(app)
    csrf.init_app(app)

    # register blueprints
    from Review.app.routes import blueprint
    app.register_blueprint(blueprint)

    # shell context for flask cli
    @app.shell_context_processor
    def ctx():
        return {'app': app, 'db': db}

    return app
