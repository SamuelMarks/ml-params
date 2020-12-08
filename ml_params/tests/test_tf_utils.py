"""
Tests for tf_utils
"""
from unittest import TestCase
from unittest.mock import patch, MagicMock, call

from ml_params.tests.utils_for_tests import unittest_main


class TestTfUtils(TestCase):
    """
    Tests whether tf_utils work
    """

    def test_get_from_tensorflow_datasets(self) -> None:
        """
        Tests whether `get_from_tensorflow_datasets` calls the right function
        """
        # if "tensorflow_datasets" not in sys.modules: sys.modules["tensorflow_datasets"] = MagicMock()
        with patch(
            "ml_params.tf_utils.common_dataset_handler", new_callable=MagicMock
        ) as common_dataset_handler, patch(
            "tensorflow_datasets.builder", new_callable=MagicMock
        ) as builder:
            import ml_params.tf_utils

            dataset_name = "dataset_name_goes_here"
            ml_params.tf_utils.get_from_tensorflow_datasets(dataset_name)

            self.assertListEqual(
                builder.call_args_list, [call(dataset_name, data_dir=None)]
            )

            self.assertEqual(builder.call_count, 1)
            self.assertEqual(common_dataset_handler.call_count, 1)


unittest_main()
