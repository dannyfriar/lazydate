from typing import List

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from lazydate.models.config import (
    DIGITS,
    MODEL_INPUT_NAME,
    MODEL_OUTPUT_NAME,
    VOCABULARY,
)
from lazydate.models.generator import DataGenerator
from lazydate.models.tf_model import lstm_encoder_decoder
from lazydate.models.vectorizer import CharVectorizer


class DateModel:
    def __init__(self):
        self.input_vectorizer = CharVectorizer(vocabulary=VOCABULARY)
        self.output_vectorizer = CharVectorizer(vocabulary=DIGITS, max_sequence_len=8)
        self.model = lstm_encoder_decoder(
            input_sequence_len=self.input_vectorizer.max_sequence_len,
            input_vocab_size=len(self.input_vectorizer.vocabulary),
            output_sequence_len=self.output_vectorizer.max_sequence_len,
            output_vocab_size=len(self.output_vectorizer.vocabulary),
        )

    def fit(
        self,
        training_examples: int = 200000,
        validation_examples: int = 10000,
        epochs: int = 10,
        patience: int = 2,
        max_queue_size: int = 20,
        workers: int = 2,
        use_multiprocessing: bool = False,
    ):
        gen_train = DataGenerator(n_examples=training_examples)
        gen_val = DataGenerator(n_examples=validation_examples)

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )

        history = self.model.fit(
            gen_train,
            epochs=epochs,
            callbacks=[early_stopping],
            validation_data=gen_val,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        return history

    def predict_on_batch(self, input_strings: List[str]) -> List[str]:
        model_inputs = {
            MODEL_INPUT_NAME: self.input_vectorizer.transform(input_strings)
        }
        pred_dict = self.model.predict(model_inputs)
        output_arrays = np.argmax(pred_dict[MODEL_OUTPUT_NAME], axis=-1)
        pred_strings = self.output_vectorizer.inverse_transform(output_arrays)
        return pred_strings

    def predict(self, input_string: str) -> str:
        return self.predict_on_batch([input_string])[0]

    def save_weights(self, fn: str):
        self.model.save_weights(fn)

    def load_weights(self, fn: str):
        self.model.load_weights(fn)
