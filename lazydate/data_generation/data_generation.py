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
    YEAR_FORMATS, ADDITIONAL_PUNCTUATION,
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

    reverse_date = np.random.rand() <= 0.3
    current_year = datetime.datetime.now().year
    if not reverse_date and current_year - 80 <= date.year <= current_year + 20:
        year_format = np.random.choice(YEAR_FORMATS)
    else:
        year_format = "yyyy"

    append_time = np.random.rand() <= 0.5
    drop_day = date.day == 1 and np.random.rand() <= 0.3
    month_format = np.random.choice(MONTH_FORMATS)
    gen_dict = {
        "day": np.random.choice(DAY_FORMATS),
        "month": month_format,
        "year": year_format,
        "separator": np.random.choice(
            possible_separators, p=list(SEPARATOR_FREQUENCY.values())
        ),
        "append_time": append_time,
        "drop_day": drop_day,
        "reverse_date": reverse_date,
    }
    gen_dict["reverse_str_month_day"] = (
        np.random.rand() <= 0.3 and len(month_format) > 2
    )

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
    elif gen_dict["reverse_str_month_day"]:
        format_date_str = (
            f"{gen_dict['month']}{sep}{gen_dict['day']}{sep}{gen_dict['year']}"
        )
    elif reverse_date:
        format_date_str = (
            f"{gen_dict['year']}{sep}{gen_dict['month']}{sep}{gen_dict['day']}"
        )
    else:
        format_date_str = (
            f"{gen_dict['day']}{sep}{gen_dict['month']}{sep}{gen_dict['year']}"
        )
    format_time_str = ""

    if append_time and not reverse_date:
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
    casing_rand_val = np.random.rand()
    casing = None
    if casing_rand_val <= 0.15:
        casing = "uppercase"
    elif casing_rand_val <= 0.3:
        casing = "lowercase"

    gen_dict = {
        "locale": np.random.choice(LOCALES),
        "append_day_suffix": append_day_suffix,
        "aug_char_action": np.random.choice(["insert", "substitute"]),
        "place_in_sentence": place_in_sentence,
        "sentence": get_random_wiki_sentence() if place_in_sentence else "",
        "casing": casing,
        "noisy_separator": np.random.random() <= 0.3,
    }

    day_suffix = ""
    if append_day_suffix and not format_dict["reverse_date"]:
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

    out = ""
    for idx, date_part in enumerate(date_parts):
        part_sep = sep
        rand_val = np.random.random()
        if noise_dict["noisy_separator"] and rand_val <= 0.15:
            part_sep += " "
        if noise_dict["noisy_separator"] and rand_val <= 0.15:
            part_sep = " " + part_sep
        elif noise_dict["noisy_separator"] and rand_val <= 0.5:
            part_sep += "".join(np.random.choice(ADDITIONAL_PUNCTUATION, size=2))

        if idx == 0:
            out += date_part
        else:
            out += f"{part_sep}{date_part}"

    # out = f"{sep}".join(date_parts)
    if noise_dict["casing"] == "uppercase":
        out = out.upper()
    elif noise_dict["casing"] == "lowercase":
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
