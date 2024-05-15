from . import error_bp
from flask import render_template

@error_bp.errorhandler(Exception)
def handle_all_exceptions(e):
    # Default error code to 500 (Internal Server Error) if the exception has no HTTP status code
    code = getattr(e, 'code', 404)
    error_message = str(e) if hasattr(e, 'description') else "Something went wrong."
    return render_template('error.html', code=code, error_message=error_message), code