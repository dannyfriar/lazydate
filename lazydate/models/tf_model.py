from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Dense,
    Embedding,
    Input,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from lazydate.models.config import MODEL_INPUT_NAME, MODEL_OUTPUT_NAME


def sequence_accuracy(y_true, y_pred):
    y_pred = K.cast(K.argmax(y_pred, axis=-1), "float32")
    diff = K.abs(y_pred - y_true)
    errors = K.sum(diff, axis=-1)
    errors = K.clip(errors, 0, 1)
    errors = K.cast(errors, K.floatx())
    return 1.0 - K.mean(errors)


def lstm_encoder_decoder(
    input_sequence_len: int,
    input_vocab_size: int,
    output_sequence_len: int,
    output_vocab_size: int,
    embedding_dim: int = 64,
    lstm_hidden_dim: int = 64,
    learning_rate: float = 1e-3,
):
    _input = Input(shape=(input_sequence_len,), name=MODEL_INPUT_NAME)
    embedding = Embedding(
        output_dim=embedding_dim, input_dim=input_vocab_size, mask_zero=False
    )(_input)
    encoding = Bidirectional(LSTM(lstm_hidden_dim, return_sequences=False))(embedding)
    repeat_encoding = RepeatVector(output_sequence_len)(encoding)

    decoding = LSTM(lstm_hidden_dim, return_sequences=True)(repeat_encoding)
    _output = TimeDistributed(Dense(output_vocab_size, activation="softmax"))(decoding)

    model = Model(inputs=[_input], outputs={MODEL_OUTPUT_NAME: _output})
    optimizer = Adam(lr=learning_rate)
    model.compile(
        optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", sequence_accuracy],
    )
    return model
