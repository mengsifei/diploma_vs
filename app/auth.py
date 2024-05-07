from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, current_user, login_required
from .models import User
from . import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['GET', 'POST'])
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
        return redirect(url_for('auth.login', message="Registration successful! Please log in."))
    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('main.dashboard'))
        else:
            return render_template('login.html', message="Invalid email or password.")
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

@auth_bp.route('/about_us')
def about_us():
    return render_template('about_us.html')
