# Modern-Keras TabCNN (Wiggins & Kim 2019 architecture, faithful) with an
# optional per-string LogSoftmax head for the ONNX export (the decoder consumes
# log-probs). Backbone outputs logits [6,21]; training uses per-string
# softmax-CE from logits; export appends LogSoftmax(axis=-1).
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_STRINGS = 6
NUM_CLASSES = 21  # 0 = closed/silent, k = fret k-1
CON_WIN = 9
N_BINS = 192


def build_backbone():
    inp = keras.Input(shape=(N_BINS, CON_WIN, 1), name="input", dtype="float32")
    x = layers.Conv2D(32, (3, 3), activation="relu")(inp)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(NUM_CLASSES * NUM_STRINGS)(x)
    logits = layers.Reshape((NUM_STRINGS, NUM_CLASSES), name="logits")(x)
    return keras.Model(inp, logits, name="tabcnn_backbone")


def per_string_loss(y_true, y_logits):
    # y_true: [B,6,21] one-hot; y_logits: [B,6,21] logits. Sum of per-string
    # softmax cross-entropy (matches the repo's catcross_by_string).
    ce = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_logits)
    return tf.reduce_sum(ce, axis=-1)


def avg_acc(y_true, y_logits):
    return tf.reduce_mean(
        tf.cast(
            tf.equal(tf.argmax(y_true, -1), tf.argmax(y_logits, -1)), tf.float32
        )
    )


def build_export_model(backbone):
    # Append a per-string LogSoftmax head (axis=-1 = the 21 classes per string).
    out = layers.Activation(
        lambda t: tf.nn.log_softmax(t, axis=-1), name="output"
    )(backbone.output)
    return keras.Model(backbone.input, out, name="tabcnn")
