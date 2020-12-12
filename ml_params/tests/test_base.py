"""
Tests for base class
"""
from inspect import getmembers
from operator import itemgetter
from unittest import TestCase
from unittest.mock import patch, MagicMock

from ml_params.base import BaseTrainer
from ml_params.tests.utils_for_tests import unittest_main
from ml_params.utils import to_d


class TestBaseTrainer(TestCase):
    """
    Tests for `BaseTrainer`
    """

    # @skip("No clue how to write this test")
    # def test_no_custom_modules(self):
    #     """
    #     Tests when modules are missing
    #     """
    #
    #     with patch.dict(sys.modules, {"os": None}), patch("sys.path", []):
    #         import ml_params.base
    #
    #         self.assertIsNone(ml_params.base.np)
    #         self.assertIsNone(ml_params.base.tf)

    def test_properties(self) -> None:
        """
        Tests whether `BaseTrainer` has the right properties
        """
        self.assertListEqual(
            sorted(to_d(BaseTrainer).keys()),
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

    def test_c_suffix(self) -> None:
        """ All '_c' suffix methods accepts a dictionary or config class instance """
        with patch.multiple(
            BaseTrainer,
            __abstractmethods__=set(),
            load_data=MagicMock,
            load_model=MagicMock,
            train=MagicMock,
        ):
            trainer = BaseTrainer()
            trainer.load_data_c(config={"dataset_name": " "})
            trainer.load_model_c(config={"model": " "})
            trainer.train_c(config={"epochs": 5})
            self.assertTupleEqual(
                tuple(
                    map(
                        itemgetter(0),
                        filter(
                            lambda name_meth: name_meth[0].endswith("_c"),
                            getmembers(BaseTrainer),
                        ),
                    )
                ),
                ("load_data_c", "load_model_c", "train_c"),
            )


unittest_main()
