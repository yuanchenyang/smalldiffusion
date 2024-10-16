import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run_slow", action="store_true", default=False, help="run tests marked with 'run_slow'"
    )

def pytest_runtest_setup(item):
    if 'run_slow' in item.keywords and not item.config.getoption("--run_slow"):
        pytest.skip("need --run_slow option to run this test")
