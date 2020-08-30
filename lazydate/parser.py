from datetime import datetime
from typing import Optional

import tensorflow as tf

from lazydate.models import DateModel

tf.config.set_visible_devices([], "GPU")

_date_model = DateModel()
_date_model.load_weights("saved_models/lstm_date_model_v02")


def parse(text: str) -> Optional[datetime]:
    datestr = _date_model.predict(text)
    if datestr == "":
        return None
    return datetime.strptime(datestr, "%Y%m%d")
