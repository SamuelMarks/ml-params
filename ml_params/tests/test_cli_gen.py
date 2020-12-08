"""
Tests for the __main__ script
"""

from unittest import TestCase, skip
from unittest.mock import MagicMock, patch
from argparse import ArgumentError
from io import StringIO

from ml_params.tests.utils_for_tests import unittest_main
from ml_params.__main__ import (
    run_main,
    _build_parser,
    ImportArgumentParser,
    get_one_arg,
)


class TestMain(TestCase):
    """
    Tests whether main works as advertised
    """

    def test_run_main(self) -> None:
        """ Tests that main will be called """

        with patch(
            "ml_params.__main__.main",
            new_callable=MagicMock,
        ) as f:
            run_main()
            self.assertEqual(f.call_count, 0)

        with patch(
            "ml_params.__main__.main",
            new_callable=MagicMock,
        ) as g, patch("ml_params.__main__.__name__", "__main__"):
            run_main()
            self.assertEqual(g.call_count, 1)

    @skip
    def test_main(self) -> None:
        """ Tests that main will be called """

        self.assertRaises(
            ArgumentError,
            lambda: str(  # ml_params_tensorflow.ml_params.doctrans_cli_gen.main(
                ["python", "bar", "foo"]
            ),
        )
        with self.assertRaises(SystemExit), patch("sys.stdout", new_callable=StringIO):
            self.assertIsNone(
                None  # ml_params_tensorflow.ml_params.doctrans_cli_gen.main(["bar", "-h"])
            )

        with patch("sys.stdout", new_callable=StringIO) as f:
            self.assertIsNone(
                None  # ml_params_tensorflow.ml_params.doctrans_cli_gen.main(["bar", "howzat"])
            )
            for attr in dir(f):
                print(attr, getattr(f, attr))

    def test__build_parser(self) -> None:
        """ Basic test for `_build_parser` """
        self.assertIsInstance(_build_parser(), ImportArgumentParser)

    def test_get_one_arg(self) -> None:
        """ Basic tests for `get_one_arg` """
        foo_cli_argv = ["--foo", "bar"]
        self.assertIsNone(get_one_arg(["-f"] + foo_cli_argv[1:], foo_cli_argv), "bar")
        self.assertEqual(get_one_arg(foo_cli_argv[:1], foo_cli_argv), "bar")
        self.assertEqual(
            get_one_arg(foo_cli_argv[:1], ["--haz", "bzr"] + foo_cli_argv), "bar"
        )

        # one = "store_true",
        # self.assertEqual(get_one_arg(one, one), one[0])


unittest_main()
