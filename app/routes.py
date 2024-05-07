from flask import Blueprint, render_template, request, redirect, url_for, session, flash, abort
from flask_login import login_required, current_user
from .models import History
from .models import ElectraScoring
from . import db
import secrets

# Initialize the Electra model scorer
electra_scorer = ElectraScoring()

main_bp = Blueprint('main', __name__)

@main_bp.route('/dashboard', defaults={'page': 1})
@main_bp.route('/dashboard/page/<int:page>')
@login_required
def dashboard(page):
    page = page if page > 0 else 1
    per_page = 10
    pagination = History.query.filter_by(user_id=current_user.id).order_by(History.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    histories = pagination.items
    return render_template('dashboard.html', histories=histories, pagination=pagination)

@main_bp.route('/delete_histories', methods=['POST'])
@login_required
def delete_histories():
    history_ids = request.form.getlist('history_ids')
    History.query.filter(History.id.in_(history_ids), History.user_id == current_user.id).delete(synchronize_session=False)
    db.session.commit()
    flash('Selected histories deleted.')
    return redirect(url_for('main.dashboard'))

@main_bp.route('/generate_token')
@login_required
def generate_token():
    """Generate a new session-based token for secure URL access."""
    token = secrets.token_urlsafe(16)
    session['inference_token'] = token
    return redirect(url_for('main.infer', token=token))

@main_bp.route('/infer', methods=['GET'])
@login_required
def infer():
    token = request.args.get('token')
    if token != session.get('inference_token'):
        abort(403)

    scores = session.get('scores', [])
    return render_template('infer.html', result=scores)

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['topic'].strip()
        essay = request.form['essay'].strip()

        if not topic or not essay:
            flash('Please provide both a topic and an essay text.')
            return render_template('index.html', isempty='True')

        # Use the Electra model to score the essay
        scores = electra_scorer.score_essay(topic, essay)
        session['scores'] = scores
        if current_user.is_authenticated:
            new_history = History(
                user_id=current_user.id,
                topic=topic,
                essay=essay,
                score_tr=scores[0],
                score_cc=scores[1],
                score_lr=scores[2],
                score_gra=scores[3]
            )
            db.session.add(new_history)
            db.session.commit()
        return redirect(url_for('main.infer'))

    return render_template('index.html', isempty='False', username=current_user.nickname if current_user.is_authenticated else '')
