from flask import Flask, request, render_template, redirect, url_for, session, flash
import re
import torch
from models.electra_baseline import BaseModel
from utils.test import score_essay_vanilla
from transformers import ElectraTokenizer
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime


model = BaseModel()
model.load_state_dict(torch.load('checkpoints/best_model_electra_8_11_AdamW.pth', map_location=torch.device('cpu')))
model.eval()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from flask_login import LoginManager

login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.unauthorized_handler
def handle_unauthorized():
    return render_template('error.html', error_message='You need to be logged in to view this page.'), 401


db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    nickname = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    topic = db.Column(db.String(255), nullable=False)
    essay = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    score_tr = db.Column(db.Integer, nullable=False)
    score_cc = db.Column(db.Integer, nullable=False)
    score_lr = db.Column(db.Integer, nullable=False)
    score_gra = db.Column(db.Integer, nullable=False)
    user = db.relationship('User', backref=db.backref('histories', lazy=True))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.errorhandler(401)
def unauthorized(error):
    return render_template('error.html', error_message='You are not authorized to view this page. Please log in.'), 401

@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', error_message='This page could not be found.'), 404

@app.errorhandler(500)
def internal_server_error(error):
    return render_template('error.html', error_message='Internal server error.'), 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        nickname = request.form['nickname']
        password = request.form['password']

        existing_email = User.query.filter_by(email=email).first()
        existing_nickname = User.query.filter_by(nickname=nickname).first()

        if existing_email:
            return render_template('register.html', message="Email already registered.")
        if existing_nickname:
            return render_template('register.html', message="Nickname already taken.")

        if len(password) < 8:
            return render_template('register.html', message="Password must be at least 8 characters long.")

        new_user = User(email=email, nickname=nickname)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login', message="Registration successful! Please log in."))
    return render_template('register.html')




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', message="Invalid email or password.")
    return render_template('login.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')


@app.route('/dashboard', defaults={'page': 1})
@app.route('/dashboard/page/<int:page>')
@login_required
def dashboard(page):
    page = page if page > 0 else 1
    per_page = 10
    pagination = History.query.filter_by(user_id=current_user.id).order_by(History.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    histories = pagination.items
    return render_template('dashboard.html', histories=histories, pagination=pagination)


@app.route('/delete_histories', methods=['POST'])
@login_required
def delete_histories():
    history_ids = request.form.getlist('history_ids')
    History.query.filter(History.id.in_(history_ids), History.user_id == current_user.id).delete(synchronize_session=False)
    db.session.commit()
    flash('Selected histories deleted.')
    return redirect(url_for('dashboard'))



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['topic'].strip()
        essay = request.form['essay'].strip()

        if not topic or not essay:
            flash('Please provide both a topic and an essay text.')
            return render_template('index.html', isempty='True')

        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        scores = score_essay_vanilla(topic, essay, tokenizer, model, 'cpu')
        session['scores'] = scores.tolist()
        if current_user.is_authenticated:
            scores = [int(score) for score in scores]  # Convert scores to integers if they are not
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
        return redirect(url_for('infer'))

    return render_template('index.html', isempty='False', username=current_user.nickname if current_user.is_authenticated else '')



@app.route('/infer', methods=['GET'])
def infer():
    scores = session.get('scores', [])  # Retrieve scores from session
    return render_template('infer.html', result=scores)

with app.app_context():
    # db.drop_all()
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
