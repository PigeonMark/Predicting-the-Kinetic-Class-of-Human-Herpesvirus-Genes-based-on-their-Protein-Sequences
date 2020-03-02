from flask import Blueprint, render_template, redirect, url_for
from Review.app.src import DebugInfo
blueprint = Blueprint('review', __name__)


@blueprint.route('/')
@blueprint.route('/index')
def index():

    debug_info = DebugInfo('HSV_1', 'd3ypd5_p06486_us10')
    print(debug_info.all_names)
    print(debug_info.percentages)
    print(debug_info.scores)
    print(debug_info.context)

    return render_template('index.html', debug_info=debug_info)


