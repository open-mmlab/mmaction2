# Copyright (c) OpenMMLab. All rights reserved.
import ctypes
import random
import string


def get_random_string(length: int = 15) -> str:
    """Get random string with letters and digits.

    Args:
        length (int): Length of random string. Defaults to 15.
    """
    return ''.join(
        random.choice(string.ascii_letters + string.digits)
        for _ in range(length))


def get_thread_id() -> int:
    """Get current thread id."""
    # use ctype to find thread id
    thread_id = ctypes.CDLL('libc.so.6').syscall(186)
    return thread_id


def get_shm_dir() -> str:
    """Get shm dir for temporary usage."""
    return '/dev/shm'
