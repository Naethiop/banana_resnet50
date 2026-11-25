"""Production WSGI entry point for Gunicorn and hosted deployments."""
from app.app import create_app

app = create_app()
