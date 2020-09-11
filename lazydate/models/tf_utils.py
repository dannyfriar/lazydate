import functools

import tensorflow as tf


def use_cpu(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with tf.device(f"/CPU:0"):
            result = function(*args, **kwargs)
        return result

    return wrapper
