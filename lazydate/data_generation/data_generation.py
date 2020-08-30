import datetime
from typing import Any, Dict, Tuple

import nlpaug.augmenter.char as nac
import numpy as np
from babel.dates import format_datetime
from nltk.tokenize import sent_tokenize

from lazydate.data_generation.config import (
    DAY_FORMATS,
    HOUR_FORMATS,
    LOCALES,
    MINUTE_FORMATS,
    MONTH_FORMATS,
    SECOND_FORMATS,
    SEPARATOR_FREQUENCY,
    TIME_SEPARATORS,
    TIMEZONE_FORMATS,
    WIKIDATA_LOC,
    YEAR_FORMATS,
)

wiki_sentences = None


def load_wikidata_sentences(n_sentences: int = 10000000):
    global wiki_sentences
    if not wiki_sentences:
        with open(WIKIDATA_LOC, "r") as f:
            wikitext = f.read()
        wiki_sentences = sent_tokenize(wikitext[:n_sentences])
    return wiki_sentences


def random_date(n_years: int = 200) -> Tuple[datetime.datetime, Dict[str, int]]:
    start_date = datetime.datetime(1900, 1, 1, 0, 0, 0)
    gen_dict = {
        "days": np.random.randint(0, n_years * 365),
        "hours": np.random.randint(0, 24),
        "minutes": np.random.randint(0, 60),
        "seconds": np.random.randint(0, 60),
    }

    date = start_date + datetime.timedelta(**gen_dict)
    return date, gen_dict


def random_format(date: datetime.datetime) -> Tuple[str, Dict[str, str]]:
    possible_separators = list(SEPARATOR_FREQUENCY.keys())

    if date.year >= datetime.datetime.now().year + 1:
        year_format = np.random.choice(YEAR_FORMATS)
    else:
        year_format = "yyyy"

    append_time = np.random.rand() <= 0.5
    drop_day = date.day == 1 and np.random.rand() <= 0.3
    gen_dict = {
        "day": np.random.choice(DAY_FORMATS),
        "month": np.random.choice(MONTH_FORMATS),
        "year": year_format,
        "separator": np.random.choice(
            possible_separators, p=list(SEPARATOR_FREQUENCY.values())
        ),
        "append_time": append_time,
        "drop_day": drop_day,
    }
    if append_time:
        time_gen_dict = {
            "second": np.random.choice(SECOND_FORMATS),
            "minute": np.random.choice(MINUTE_FORMATS),
            "hour": np.random.choice(HOUR_FORMATS),
            "timezone": np.random.choice(TIMEZONE_FORMATS),
            "time_separator": np.random.choice(TIME_SEPARATORS),
        }
    else:
        time_gen_dict = {
            k: "" for k in ["second", "minute", "hours", "timezone", "time_separator"]
        }
    gen_dict.update(time_gen_dict)

    sep = gen_dict["separator"]
    if sep != "''" and gen_dict["year"] == "yy":
        if np.random.random() <= 0.5:
            gen_dict["year"] = "''" + gen_dict["year"]

    if drop_day:
        format_date_str = f"{gen_dict['month']}{sep}{gen_dict['year']}"
    else:
        format_date_str = (
            f"{gen_dict['day']}{sep}{gen_dict['month']}{sep}{gen_dict['year']}"
        )
    format_time_str = ""

    if append_time:
        sep = gen_dict["time_separator"]
        format_time_str = f" {gen_dict['hour']}{sep}{gen_dict['minute']}"
        if np.random.random() <= 0.5:
            format_time_str += f"{sep}{gen_dict['second']}"
        if np.random.random() <= 0.5:
            format_time_str += f" a"  # AM / PM
        if np.random.random() <= 0.5:
            format_time_str += f" {gen_dict['timezone']}"

    format_str = format_date_str + format_time_str
    gen_dict["format_str"] = format_str
    return format_str, gen_dict


def get_random_wiki_sentence(max_length: int = 150) -> str:
    wiki_sentences = load_wikidata_sentences()
    idx = np.random.randint(0, len(wiki_sentences))
    return wiki_sentences[idx][:max_length]


def random_noise_dict(
    date: datetime.datetime, format_dict: Dict[str, str]
) -> Dict[str, str]:
    append_day_suffix = format_dict["day"] == "dd" and np.random.random() <= 0.5
    place_in_sentence = np.random.random() <= 0.5

    # TODO: add noise to end of string without separator

    gen_dict = {
        "locale": np.random.choice(LOCALES),
        "append_day_suffix": append_day_suffix,
        "aug_char_action": np.random.choice(["insert", "substitute"]),
        "place_in_sentence": place_in_sentence,
        "sentence": get_random_wiki_sentence() if place_in_sentence else "",
        "lowercase": np.random.random() < 0.2,
    }

    day_suffix = ""
    if append_day_suffix:
        if date.day in [1, 21, 31]:
            day_suffix = "st"
        elif date.day in [2, 22]:
            day_suffix = "st"
        elif date.day in [3, 23]:
            day_suffix = "rd"
        else:
            day_suffix = "th"
    gen_dict["day_suffix"] = day_suffix

    return gen_dict


def put_datestr_in_sentence(datestr: str, sentence: str):
    split_sentence = sentence.split(" ")
    idx = np.random.randint(0, len(split_sentence))
    split_sentence[idx] = datestr
    return " ".join(split_sentence)


def apply_noise(
    datestr: str, format_dict: Dict[str, str], noise_dict: Dict[str, Any]
) -> str:
    sep = format_dict["separator"]
    sep = sep[0] if len(sep) > 1 else sep
    date_parts = datestr.split(sep)

    if noise_dict["append_day_suffix"]:
        date_parts[0] = date_parts[0] + noise_dict["day_suffix"]

    # Add spelling mistake to month name
    if len(format_dict["month"]) > 2 and np.random.random() <= 0.3:
        aug = nac.RandomCharAug(
            action=noise_dict["aug_char_action"],
            aug_char_min=1,
            aug_char_max=1,
        )
        date_parts[1] = aug.augment(date_parts[1])

    out = f"{sep}".join(date_parts)

    if noise_dict["lowercase"]:
        out = out.lower()

    if noise_dict["place_in_sentence"]:
        out = put_datestr_in_sentence(out, noise_dict["sentence"])

    return out


def generate_date(
    no_date_prob: float = 0.1,
) -> Tuple[str, datetime.datetime, Dict[str, Any]]:
    date, date_gen_dict = random_date()
    format_str, format_gen_dict = random_format(date)
    noise_gen_dict = random_noise_dict(date, format_gen_dict)

    datestr = format_datetime(
        date,
        format=format_str,
        locale=noise_gen_dict["locale"],
    )
    datestr = apply_noise(datestr, format_gen_dict, noise_gen_dict)

    gen_dict = date_gen_dict
    gen_dict.update(format_gen_dict)
    gen_dict.update(noise_gen_dict)
    gen_dict["no_date"] = False

    # Example with no date
    if np.random.random() <= no_date_prob:
        date = None
        datestr = get_random_wiki_sentence()
        gen_dict["no_date"] = True

    return datestr, date, gen_dict
