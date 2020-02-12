
import pytest
import requests
from dotenv import load_dotenv

import pdf_trio
from pdf_trio import create_app

@pytest.fixture
def flask_client():
    load_dotenv(dotenv_path="./example.env")
    app = create_app()
    app.testing = True
    app.debug = False
    app.config['WTF_CSRF_ENABLED'] = False
    return app.test_client()


def skip_if_no_tensorflow():
    """
    Raises a pytest skip exception if back-end tensorflow-serving instances are
    not available.

    Use this to have "live" tests skip if back-end not available (eg, in CI, or
    tests which don't mock the backend)
    """
    try:
        requests.get('http://localhost:8501/')
    except requests.exceptions.ConnectionError:
        pytest.skip("backend tensorflow-serving not available")
