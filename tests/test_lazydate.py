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
    ("12,08,1905", datetime(1905, 8, 12)),
]


noisy_dates = [
    ("20./n0vembr/2020", datetime(2020, 11, 20, 0, 0)),
    # ("1907-!£Sep-13", datetime(1907, 9, 13)),
]


surrounding_text = [
    ("the date is 12th nov 1982", datetime(1982, 11, 12)),
    ("the date is 12th nov 1982 after which this happened", datetime(1982, 11, 12)),
    ("lkererer 19 december '20 elrkererd", datetime(2020, 12, 19, 0, 0)),
    (
        "The Original Amateur Hour , which began on radio in the 1930s under original 1/04/ @43",
        datetime(1943, 4, 1),
    ),
    (
        "There is , however , a danger associated with any heavy gas in large quantities : it may sit invisibly in a container , and if 02-01-&@1907 2:24:28 am person enters a conta",
        datetime(1907, 1, 2),
    ),
    # (
    #     "The residence also became known as the Spencer House after Pitman sold it to his business 2005.J$ne. 12 Captain Thomas Spencer .",
    #     datetime(2005, 6, 12),
    # ),
]


no_dates = [
    ("lazydate", None),
    ("Subsequently , the system moved over the Yucatán Peninsula .", None),
    ("In 1860 he was chosen president of the American Temperance Union .", None),
]

test_dates = clean_dates + noisy_dates + surrounding_text


@pytest.mark.parametrize("datestr, datetime", test_dates)
def test_date_parsing(datestr, datetime):
    assert ld.parse(datestr) == datetime
