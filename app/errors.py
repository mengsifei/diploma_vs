from flask import Blueprint, render_template

errors_bp = Blueprint('errors', __name__)

@errors_bp.app_errorhandler(401)
def unauthorized(error):
    return render_template('error.html', error_message='You are not authorized to view this page. Please log in.'), 401

@errors_bp.app_errorhandler(403)
def forbidden(error):
    return render_template('error.html', error_message='Access forbidden: You do not have permission to view this page.'), 403

@errors_bp.app_errorhandler(404)
def page_not_found(error):
    return render_template('error.html', error_message='Page not found.'), 404

@errors_bp.app_errorhandler(500)
def internal_server_error(error):
    return render_template('error.html', error_message='Internal server error.'), 500
