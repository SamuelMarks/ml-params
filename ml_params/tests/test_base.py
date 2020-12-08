"""
Tests for base class
"""
from unittest import TestCase

from ml_params.base import BaseTrainer
from ml_params.tests.utils_for_tests import unittest_main
from ml_params.utils import to_d


class TestBaseTrainer(TestCase):
    """
    Tests for `BaseTrainer`
    """

    def test_properties(self) -> None:
        """
        Tests whether `BaseTrainer` has the right properties
        """
        self.assertListEqual(
            list(to_d(BaseTrainer).keys()),
            [
                "data",
                "load_data",
                "load_data_c",
                "load_model",
                "load_model_c",
                "model",
                "train",
                "train_c",
            ],
        )


unittest_main()
