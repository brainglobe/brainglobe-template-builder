import datetime

import fancylog
import pytest


@pytest.fixture()
def mock_fancylog_datetime(mocker):
    """Mock datetime.now for fancylog to 2025-12-10 15:15.

    This allows the log filename timestamp to remain consistent
    for testing.
    """
    mocker.patch("fancylog.fancylog.datetime")
    fancylog.fancylog.datetime.now.return_value = datetime.datetime(
        2025, 12, 10, 15, 15
    )
