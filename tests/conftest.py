import pytest


@pytest.fixture(scope="session")
def random_state():
    return 42
