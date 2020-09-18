import pytest
from datetime import datetime

import lazydate as ld
from lazydate import __version__


def test_version():
    assert __version__ == "0.1.0"


test_date_parsing = [
    ("08/12/20", datetime(2020, 12, 8)),
    ("08TH 12 2020", datetime(2020, 12, 8)),
    ("8 dec 20", datetime(2020, 12, 8)),
    ("8th december '20", datetime(2020, 12, 8)),

    ("the date is 12th nov 1982", datetime(1982, 11, 12)),
    ("the date is 12th nov 1982 after which this happened", datetime(1982, 11, 12)),

    ("22 aug 93", datetime(1993, 8, 22, 0, 0)),
    ("lkererer 19 december '20 elrkererd", datetime(2020, 12, 19, 0, 0)),
    ("20./n0vembr/2020", datetime(2020, 11, 20, 0, 0)),
]


@pytest.mark.parametrize("datestr, datetime", test_date_parsing)
def test_date_parsing(datestr, datetime):
    assert ld.parse(datestr) == datetime
