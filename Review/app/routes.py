from flask import Blueprint, render_template, redirect, url_for, request
from Review.app.src import paper_url, GeneRotator, sorted_keys_phases
from Review.app.models import Gene
from Review.app.forms import DeleteReviewForm, AddReviewForm

blueprint = Blueprint('review', __name__)


@blueprint.route('/', methods=['GET', 'POST'])
@blueprint.route('/index', methods=['GET', 'POST'])
def index():
    # Setup gene rotator first time
    if GeneRotator.current is None:
        GeneRotator.next()

    (virus, gene, gene_debug_info), paper_titles = GeneRotator.get()
    if virus is not None:
        add_review_form = AddReviewForm()

        if add_review_form.validate_on_submit() and add_review_form.submit_add_review.data is True:
            old_phase = gene_debug_info.winning_phase()

            if add_review_form.status.data == 'CORRECT':
                Gene.add(virus, gene, old_phase, add_review_form.status.data, old_phase)
                GeneRotator.next()
                return redirect(url_for('review.index'))
            elif add_review_form.status.data == 'UNCERTAIN' or add_review_form.status.data == 'REVIEW_LATER':
                Gene.add(virus, gene, old_phase, add_review_form.status.data)
                GeneRotator.next()
                return redirect(url_for('review.index'))

            elif add_review_form.status.data == 'MODIFIED':
                Gene.add(virus, gene, old_phase, add_review_form.status.data, add_review_form.phase.data)
                GeneRotator.next()
                return redirect(url_for('review.index'))

        return render_template('index.html', debug_info=gene_debug_info, paper_titles=paper_titles, round=round,
                               paper_url=paper_url, str=str, add_review_form=add_review_form,
                               sorted_keys_phases=sorted_keys_phases)

    else:
        return render_template('all_done.html', totals=Gene.get_totals())


@blueprint.route('/overview', methods=['GET', 'POST'])
def overview():
    delete_review_form = DeleteReviewForm()
    if delete_review_form.validate_on_submit() and delete_review_form.submit_delete_review.data is True:
        Gene.delete(delete_review_form.names.data)
        return redirect(url_for('review.overview'))

    sort = False
    if 'sort' in request.args:
        if request.args['sort'] == 'True':
            sort = True

    return render_template('overview.html', genes=Gene.get_all(sort), delete_review_form=delete_review_form, none=None,
                           debug_info=GeneRotator.debug_info, totals=Gene.get_totals())


@blueprint.route('/index/<virus>/<gene>', methods=['GET', 'POST'])
def single_gene(virus, gene):
    # Setup gene rotator first time
    if GeneRotator.current is None:
        GeneRotator.next()

    gene_debug_info = GeneRotator.debug_info[virus][gene]
    paper_titles = GeneRotator.paper_titles

    add_review_form = AddReviewForm()

    overview_sorted = False
    if 'overview_sorted' in request.args:
        if request.args['overview_sorted'] == 'True':
            overview_sorted = True

    if add_review_form.validate_on_submit() and add_review_form.submit_add_review.data is True:

        old_phase = gene_debug_info.winning_phase()

        if add_review_form.status.data == 'CORRECT':
            Gene.update(gene, add_review_form.status.data, old_phase)
            GeneRotator.next()
            return redirect(url_for('review.overview', sort=overview_sorted))
        elif add_review_form.status.data == 'UNCERTAIN' or add_review_form.status.data == 'REVIEW_LATER':
            Gene.update(gene, add_review_form.status.data, '')
            GeneRotator.next()
            return redirect(url_for('review.overview', sort=overview_sorted))

        elif add_review_form.status.data == 'MODIFIED':
            Gene.update(gene, add_review_form.status.data, add_review_form.phase.data)
            GeneRotator.next()
            return redirect(url_for('review.overview', sort=overview_sorted))

    return render_template('index.html', debug_info=gene_debug_info, paper_titles=paper_titles, round=round,
                           paper_url=paper_url, str=str, add_review_form=add_review_form,
                           sorted_keys_phases=sorted_keys_phases)
# @blueprint.route('/test')
# def test():
#     return render_template('test.html')
