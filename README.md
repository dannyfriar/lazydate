## Lazydate

Small python date-parsing library to convert noisy date strings to Python `datetime.datetime` objects with an encoder-decoder model.

Designed to be robust to mis-spellings, different date formats and surrounding text.


```python
import lazydate as ld

ld.parse("22 aug 93")
>>> datetime.datetime(1993, 8, 22, 0, 0)

ld.parse("the event will happen on 19 december '20")
>>> datetime.datetime(2020, 12, 19, 0, 0)

ld.parse("20./n0vembr/2020")
>>>datetime.datetime(2020, 11, 20, 0, 0)
```

