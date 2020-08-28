import math
from typing import List, Tuple

from keras.utils import Sequence

from lazydate.data_generation import generate_date
from lazydate.model.config import VOCABULARY, DIGITS, UNK_TOKEN, MODEL_INPUT_NAME, MODEL_OUTPUT_NAME
from lazydate.model.vectorizer import CharVectorizer


class DataGenerator(Sequence):
    def __init__(self, batch_size=32, n_examples=50000):
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.input_vectorizer = CharVectorizer(vocabulary=VOCABULARY)
        self.output_vectorizer = CharVectorizer(vocabulary=DIGITS, max_sequence_len=8)

    def __len__(self):
        return int(math.ceil(self.n_examples / self.batch_size))

    @property
    def input_sequence_len(self):
        return self.input_vectorizer.max_sequence_len

    @property
    def input_vocab_size(self):
        return len(self.input_vectorizer.vocabulary)

    @property
    def output_sequence_len(self):
        return self.output_vectorizer.max_sequence_len

    @property
    def output_vocab_size(self):
        return len(self.output_vectorizer.vocabulary)

    def generate_string_batch(self) -> Tuple[List[str], List[str]]:
        input_strings: List[str] = []
        output_strings: List[str] = []

        for _ in range(self.batch_size):
            datestr, date, gen_dict = generate_date()
            if date:
                output_datestr = date.strftime("%Y%m%d")
            else:
                output_datestr = "".join([UNK_TOKEN] * self.output_vectorizer.max_sequence_len)
            input_strings.append(datestr)
            output_strings.append(output_datestr)

        return input_strings, output_strings

    def __getitem__(self, idx: int):
        input_strings, output_strings = self.generate_string_batch()
        inputs = {MODEL_INPUT_NAME: self.input_vectorizer.transform(input_strings)}
        outputs = {MODEL_OUTPUT_NAME: self.output_vectorizer.transform(output_strings)}
        return inputs, outputs
