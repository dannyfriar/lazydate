import logging
from datetime import datetime
from typing import List, Optional

from lazydate.models import DateModel
from lazydate.models.config import MAX_SEQUENCE_LEN
from lazydate.models.tf_utils import use_cpu

logger = logging.getLogger(__name__)
_date_model = None


def _load_date_model():
    global _date_model
    if not _date_model:
        _date_model = DateModel()
        _date_model.load_weights("saved_models/lstm_date_model_v05")
    return _date_model


@use_cpu
def parse(text: str) -> Optional[datetime]:
    if len(text) > MAX_SEQUENCE_LEN:
        logger.warning(
            "Input to lazydate.parse is longer than max sequence length - "
            f"{len(text)} > {MAX_SEQUENCE_LEN} - input will be truncated"
        )

    date_model = _load_date_model()
    datestr = date_model.predict(text)
    if datestr == "":
        return None
    return datetime.strptime(datestr, "%Y%m%d")


@use_cpu
def parse_batch(texts: List[str]) -> List[Optional[datetime]]:
    if len(texts) == 0:
        return []

    max_length = max([len(t) for t in texts])
    if max_length > MAX_SEQUENCE_LEN:
        logger.warning(
            "Longest input to lazydate.parse_batch is longer than max sequence length - "
            f"{max_length} > {MAX_SEQUENCE_LEN} - input will be truncated"
        )

    date_model = _load_date_model()
    datestrs = date_model.predict_on_batch(texts)
    dates = [datetime.strptime(d, "%Y%m%d") if d != "" else None for d in datestrs]
    return dates
