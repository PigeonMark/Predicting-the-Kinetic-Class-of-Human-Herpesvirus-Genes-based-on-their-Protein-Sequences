from flask import Blueprint, render_template, redirect, url_for
from Review.app.src import DebugInfo, paper_url, paper_title

blueprint = Blueprint('review', __name__)


@blueprint.route('/')
@blueprint.route('/index')
def index():
    debug_info = DebugInfo('HSV_1', 'p10215_ul31')
    return render_template('index.html', debug_info=debug_info, round=round, paper_url=paper_url,
                           paper_title=paper_title, str=str)
