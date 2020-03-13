import json

from sqlalchemy.sql import exists
from sqlalchemy import exc, desc
from Review.app import db

REVIEW_STATUSES = {"CORRECT", "MODIFIED", "UNCERTAIN", "REVIEW_LATER"}
with open("config/general.json", 'r') as general_config_file:
    general_config = json.load(general_config_file)
    REVIEWED_PHASES = list(general_config['phases'].keys()) + ['']


class Gene(db.Model):
    virus = db.Column(db.String, nullable=False)
    names = db.Column(db.String, primary_key=True, nullable=False)
    old_phase = db.Column(db.String, nullable=False)
    reviewed_phase = db.Column(db.String)
    review_status = db.Column(db.String, nullable=False)

    def __repr__(self):
        return f"Names: {self.names}\nReviewed phase: {self.reviewed_phase}\nReview status: {self.review_status}"

    @staticmethod
    def exists(names):
        return db.session.query(exists().where(Gene.names == names)).scalar()

    @staticmethod
    def add(virus, names, old_phase, review_status, reviewed_phase=None):

        if Gene.exists(names):
            raise Exception(f"Error in Gene.add, gene with name {names} already exists")

        if review_status is None or review_status not in REVIEW_STATUSES:
            raise Exception(f"Unknown review status: {review_status}")
        if reviewed_phase is not None and reviewed_phase not in REVIEWED_PHASES:
            raise Exception(f"Unknown reviewed phase: {reviewed_phase}")
        if virus is None:
            raise Exception(f"Virus can't be None")
        if old_phase is None or old_phase not in REVIEWED_PHASES:
            raise Exception(f"Unknown old phase: {old_phase}")

        try:
            new_gene = Gene(virus=virus, names=names, old_phase=old_phase, reviewed_phase=reviewed_phase,
                            review_status=review_status)
            db.session.add(new_gene)
            db.session.commit()
            return True

        except exc.IntegrityError:
            db.session.rollback()
            print(f'Database error in Gene.add with {new_gene}')
            return False

    @staticmethod
    def get_by_names(names):
        return Gene.query.filter_by(names=names).first()

    @staticmethod
    def update(names, review_status=None, reviewed_phase=None):
        if not Gene.exists(names):
            raise Exception(f"Unkown gene to update: {names}")
        if review_status is not None and review_status not in REVIEW_STATUSES:
            raise Exception(f"Unknown review status: {review_status}")
        if reviewed_phase is not None and reviewed_phase not in REVIEWED_PHASES:
            raise Exception(f"Unknown reviewed phase: {reviewed_phase}")

        gene = Gene.get_by_names(names)

        try:
            if review_status is not None:
                gene.review_status = review_status
            if reviewed_phase is not None:
                gene.reviewed_phase = reviewed_phase
            db.session.commit()
            return True

        except exc.IntegrityError:
            db.session.rollback()
            print(f'Database error in Gene.update with {gene}')
            return False

    @staticmethod
    def delete(names):
        if not Gene.exists(names):
            raise Exception(f"Unkown gene to delete: {names}")

        gene = Gene.get_by_names(names)

        try:
            db.session.delete(gene)
            db.session.commit()
            return True
        except exc.IntegrityError:
            db.session.rollack()
            print(f'Database error in Gene.delete with {gene}')
            return False

    @staticmethod
    def get_all(sort=False):
        all_genes = Gene.query.all()
        if sort:
            SORT_ORDER = {"CORRECT": 3, "MODIFIED": 2, "UNCERTAIN": 1, "REVIEW_LATER": 0}
            sorted_genes = sorted(all_genes, key=lambda gene: SORT_ORDER[gene.review_status])
            return sorted_genes
        else:
            return reversed(all_genes)

    @staticmethod
    def get_totals():
        from Review.app.src import GeneRotator

        totals = {}
        statuses = {}
        for gene in Gene.get_all():
            if gene.virus not in totals:
                totals[gene.virus] = 1
            else:
                totals[gene.virus] += 1

            if gene.review_status not in statuses:
                statuses[gene.review_status] = 1
            else:
                statuses[gene.review_status] += 1

        to_review_totals = {}
        for virus, genes in GeneRotator.debug_info.items():
            to_review_totals[virus] = len(genes)

        return totals, to_review_totals, sum([t for v, t in totals.items()]), sum([t for v, t in to_review_totals.items()]), statuses
