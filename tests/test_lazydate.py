import pytest
from datetime import datetime

import lazydate as ld
from lazydate import __version__


def test_version():
    assert __version__ == "0.1.0"


clean_dates = [
    ("08/12/20", datetime(2020, 12, 8)),
    ("08TH 12 2020", datetime(2020, 12, 8)),
    ("8 dec 20", datetime(2020, 12, 8)),
    ("8th december '20", datetime(2020, 12, 8)),
    ("22 aug 93", datetime(1993, 8, 22, 0, 0)),
]


noisy_dates = [
    ("20./n0vembr/2020", datetime(2020, 11, 20, 0, 0)),
]


surrounding_text = [
    ("the date is 12th nov 1982", datetime(1982, 11, 12)),
    ("the date is 12th nov 1982 after which this happened", datetime(1982, 11, 12)),
    ("lkererer 19 december '20 elrkererd", datetime(2020, 12, 19, 0, 0)),
    (
        "The Original Amateur Hour , which began on radio in the 1930s under original 1/04/ @43",
        datetime(1943, 4, 1),
    ),
]

test_dates = clean_dates + noisy_dates + surrounding_text


@pytest.mark.parametrize("datestr, datetime", test_dates)
def test_date_parsing(datestr, datetime):
    assert ld.parse(datestr) == datetime
