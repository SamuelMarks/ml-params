"""
Shared utility functions used by tests
"""
from os import name
from sys import version_info
from unittest import main

# Windows is support for TensorFlow, but tfds needs my https://github.com/tensorflow/datasets/pull/2840
TF_SUPPORTED = (3, 8) >= version_info[:2] > (3, 5) and name != "nt"


def unittest_main():
    """ Runs unittest.main if __main__ """
    if __name__ == "__main__":
        main()


# From https://github.com/Suor/funcy/blob/0ee7ae8/funcy/funcs.py#L34-L36
def rpartial(func, *args):
    """Partially applies last arguments."""
    return lambda *a: func(*(a + args))


__all__ = ["unittest_main", "rpartial", "TF_SUPPORTED"]
