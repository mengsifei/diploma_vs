from flask import Flask, request, render_template, redirect, url_for, session, flash
import torch
from models.electra_baseline import BaseModel
from utils.test import score_essay_vanilla
from transformers import ElectraTokenizer
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash


model = BaseModel()
model.load_state_dict(torch.load('checkpoints/best_model_electra_8_11_AdamW.pth', map_location=torch.device('cpu')))
model.eval()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Database model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    nickname = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        nickname = request.form['nickname']
        password = request.form['password']

        user = User(email=email, nickname=nickname)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        login_user(user)  # Log in the user immediately after registering
        return redirect(url_for('index'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')

    return render_template('login.html')



@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['topic']
        essay = request.form['essay']
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        scores = score_essay_vanilla(topic, essay, tokenizer, model, 'cpu')
        session['scores'] = scores.tolist()  # Store scores in session
        return redirect(url_for('infer'))
    return render_template('index.html')

@app.route('/infer', methods=['GET'])
def infer():
    scores = session.get('scores', [])  # Retrieve scores from session
    return render_template('infer.html', result=scores)

if __name__ == '__main__':
    app.run(debug=True)
