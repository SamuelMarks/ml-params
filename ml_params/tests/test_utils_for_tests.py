"""
Tests for utils for tests
"""
from io import StringIO
from unittest import TestCase
from unittest.mock import patch, MagicMock

from ml_params.tests.utils_for_tests import unittest_main, rpartial


class TestUtilsForTests(TestCase):
    """
    Tests whether utils for tests work
    """

    def test_unittest_main(self) -> None:
        """
        Tests whether `unittest_main` is called when `__name__ == '__main__'`
        """
        self.assertEqual(type(unittest_main).__name__, "function")
        self.assertIsNone(unittest_main())
        argparse_mock = MagicMock()
        with patch("ml_params.tests.utils_for_tests.__name__", "__main__"), patch(
            "sys.stderr", new_callable=StringIO
        ), self.assertRaises(SystemExit) as e:
            import ml_params.tests.utils_for_tests

            ml_params.tests.utils_for_tests.unittest_main()

        self.assertIsInstance(e.exception.code, bool)
        self.assertIsNone(argparse_mock.call_args)
        self.assertIsNone(ml_params.tests.utils_for_tests.unittest_main())

    def test_rpartial(self) -> None:
        """ Test that rpartial works as advertised """
        self.assertTrue(rpartial(isinstance, str)(""))
        self.assertFalse(rpartial(isinstance, str)(0))


unittest_main()
