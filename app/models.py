from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from transformers import ElectraTokenizer
from models.electra_baseline import BaseModel
from utils.test import score_essay_vanilla
from . import db

class ElectraScoring:
    def __init__(self, model_path='checkpoints/best_model_electra_8_11_AdamW.pth'):
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
        self.model = BaseModel()
        # self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def score_essay(self, topic, essay):
        """Perform scoring using the loaded model."""
        scores = score_essay_vanilla(topic, essay, self.tokenizer, self.model, 'cpu')
        return scores.tolist()

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
