from typing import List, Dict

import numpy as np

from lazydate.models.config import UNK_TOKEN, MAX_SEQUENCE_LEN


class CharVectorizer:
    def __init__(self, vocabulary: str, max_sequence_len: int = MAX_SEQUENCE_LEN):
        self.max_sequence_len = max_sequence_len
        self.encoder: Dict[str, int] = {
            c: idx for idx, c in enumerate(list(vocabulary))
        }
        self.encoder[UNK_TOKEN] = len(self.encoder)

    @property
    def decoder(self):
        return {v: k for k, v in self.encoder.items()}

    @property
    def vocabulary(self):
        return sorted(list(self.encoder.keys()))

    def transform(self, inputs: List[str]) -> np.ndarray:
        outputs = [self._get_char_indices_for_word(s) for s in inputs]
        outputs = np.array(outputs)
        return outputs

    def inverse_transform(self, arr: np.ndarray) -> List[str]:
        """
        :param arr: (n_examples, self.max_sequence_length)
        :return: List[str]
        """
        output_strings: List[str] = []

        for output in arr:
            decoded_output = [self.decoder[o] for o in output]
            if "<unk>" in decoded_output:
                output_strings.append("")
            else:
                output_strings.append("".join(decoded_output))

        return output_strings

    def _get_char_indices_for_word(self, text: str) -> np.ndarray:
        next_arr = np.zeros([self.max_sequence_len], dtype=np.int32)

        for idx, token in enumerate(text):
            if idx < self.max_sequence_len:  # truncate end of sentence if too long
                if token in self.encoder:
                    vocab_idx = self.encoder[token]
                else:
                    vocab_idx = self.encoder[UNK_TOKEN]
                next_arr[idx] = vocab_idx
        return next_arr
