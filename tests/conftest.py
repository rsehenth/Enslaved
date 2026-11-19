import pytest
from fastapi.testclient import TestClient

from backend.core.directives import Directives, load_directives
from backend.main import create_app
from backend.providers.simulated import SimulatedProvider
from backend.storage.database import Database
from tests.helpers import make_settings


@pytest.fixture
def settings(tmp_path):
    return make_settings(tmp_path)


@pytest.fixture
def directives(settings):
    return load_directives(settings.directives.path)


@pytest.fixture
def db(settings):
    database = Database(settings.storage.database_path)
    database.connect()
    yield database
    database.close()


@pytest.fixture
def provider():
    return SimulatedProvider(model="simulado-1b")


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client
