import pytest

from app.grading import Grader
from app.methods.context import RunContext
from app.mock import MockBackend
from app.schema import Sampling
from tests.helpers import QS


@pytest.fixture(scope="session")
def grader():
    g = Grader(workers=2)
    yield g
    g.close()


@pytest.fixture
def events():
    return []


@pytest.fixture
def ctx(grader, events):
    async def emit(e):
        events.append(e)
    return RunContext(backend=MockBackend(QS), grader=grader, sampling=Sampling(), emit=emit)
