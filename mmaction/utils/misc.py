import ctypes
import random
import string

import numpy as np


def get_random_string(length=15):
    """Get random string with letters and digits.

    Args:
        length (int): Length of random string. Default: 15.
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length))


def get_thread_id():
    """Get current thread id."""
    # use ctype to find thread id
    thread_id = ctypes.CDLL('libc.so.6').syscall(186)
    return thread_id


def get_shm_dir():
    """Get shm dir for temporary usage."""
    return '/dev/shm'


def softmax(x, dim=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)
