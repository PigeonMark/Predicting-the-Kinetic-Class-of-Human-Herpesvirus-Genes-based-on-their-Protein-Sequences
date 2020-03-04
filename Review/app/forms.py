from flask_wtf import FlaskForm
from wtforms import SubmitField, HiddenField, StringField


class DeleteReviewForm(FlaskForm):
    names = HiddenField()
    submit_delete_review = SubmitField('Delete')


class AddReviewForm(FlaskForm):
    status = HiddenField()
    phase = HiddenField()
    submit_add_review = SubmitField()
