import babel

DAY_FORMATS = ["d", "dd"]
MONTH_FORMATS = ["M", "MM", "MMM", "MMMM", "MMMM", "L", "LL", "LLL", "LLLL", "LLLL"]
YEAR_FORMATS = ["yy", "yyyy"]
SECOND_FORMATS = ["s", "ss"]
MINUTE_FORMATS = ["m", "mm"]
HOUR_FORMATS = ["h", "hh", "H", "HH"]
TIMEZONE_FORMATS = ["", "", "", "", "", "z", "zz", "zzz", "zzzz"]
TIME_SEPARATORS = [":"]
SEPARATOR_FREQUENCY = {
    ".": 0.1,
    "/": 0.15,
    "-": 0.15,
    "''": 0.1,
    " ": 0.5,
}
BUILT_IN_FORMATS = ["short", "medium", "long", "full"]
LOCALES = babel.localedata.locale_identifiers()
LOCALES = [l for l in LOCALES if "en_" in l]
WIKIDATA_LOC = "data/wiki.train.raw"
