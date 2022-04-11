# Copyright (c) OpenMMLab. All rights reserved.
import ctypes
import random
import string


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


def visualize_confusion_matrix(confusion_matrix):
    """Visualize a confusion matrix.

    Args:
        confusion_matrix (np.array): the confusion matrix
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(15, 10))
    sns.heatmap(confusion_matrix, annot=True, square=True, cbar=True)
    plt.show()
