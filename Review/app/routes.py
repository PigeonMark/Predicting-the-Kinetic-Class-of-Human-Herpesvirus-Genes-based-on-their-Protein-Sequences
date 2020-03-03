from flask import Blueprint, render_template, redirect, url_for
from Review.app.src import paper_url
from DebugInfoCollector import DebugInfoCollector

blueprint = Blueprint('review', __name__)

debug_info_collector = DebugInfoCollector('config/debug_info_collector_config.json')
debug_info = debug_info_collector.load_debug_info()
paper_titles = debug_info_collector.load_paper_titles()


@blueprint.route('/')
@blueprint.route('/index')
def index():
    gene_debug_info = debug_info['HSV_1']['p04487_p56958_us11']

    return render_template('index.html', debug_info=gene_debug_info, paper_titles=paper_titles, round=round,
                           paper_url=paper_url, str=str)
