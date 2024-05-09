from flask import render_template, session, request, redirect, url_for, abort, flash
from . import infer_bp
import secrets
from transformers import ElectraTokenizer
from utils.test import score_essay_vanilla
from ..models import History, db
from flask_login import current_user
from models.electra_baseline import BaseModel

@infer_bp.route('/infer', methods=['GET'])
def infer():
    """Secure inference page access with session token validation."""
    token = request.args.get('token')
    if token is None:
        flash("Please go to the home page to check your essay.")
        return redirect(url_for('infer.index'))
    # Verify that the token matches the one in the session
    if token != session.get('inference_token'):
        flash("Invalid or expired access token. Please submit the form again.")
        return redirect(url_for('infer.index'))

    # Retrieve scores, topic, and essay if the token matches
    scores = session.get('scores', [])
    topic = session.get('topic', "")
    essay = session.get('essay', "")
    # Optionally clear the session inference token after valid access
    session.pop('inference_token', None)

    return render_template('infer.html', result=scores, topic=topic, essay=essay)

@infer_bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['topic'].strip()
        essay = request.form['essay'].strip()

        if not topic or not essay:
            flash('Please provide both a topic and an essay text.')
            return render_template('index.html', isempty='True')

        # Prepare tokenizer and model
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        model = BaseModel()
        scores = score_essay_vanilla(topic, essay, tokenizer, model, 'cpu')  # Adjust for model instance
        session['scores'] = scores.tolist()
        session['topic'] = topic  # Store the topic in the session
        session['essay'] = essay  # Store the essay in the session
        
        # Generate a secure token and store it in the session
        token = secrets.token_urlsafe(16)
        session['inference_token'] = token

        # Optionally save to history if authenticated
        if current_user.is_authenticated:
            scores = [int(score) for score in scores]  # Convert scores to integers
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
        
        # Redirect to the `/infer` page with the generated token
        return redirect(url_for('infer.infer', token=token))

    return render_template('index.html', isempty='False', username=current_user.nickname if current_user.is_authenticated else '')

@infer_bp.route('/rubric_explanation')
def rubric_explanation():
    return render_template('rubric_explanation.html')
