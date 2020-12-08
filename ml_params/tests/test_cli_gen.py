"""
Tests for the __main__ script
"""

import sys
from io import StringIO
from os import environ
from unittest import TestCase, skipIf
from unittest.mock import MagicMock, patch

import ml_params
from ml_params.__main__ import (
    run_main,
    _build_parser,
    ImportArgumentParser,
    get_one_arg,
    main,
)
from ml_params.tests.utils_for_tests import unittest_main


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

    @skipIf(sys.version_info[:2] == (3, 5), "Enums are broken in 3.5?")
    def test_main(self) -> None:
        """ Tests that main will be called """

        with self.assertRaises(SystemExit) as e, patch(
            "sys.stdout", new_callable=StringIO
        ) as out, patch("sys.stderr", new_callable=StringIO) as err, patch(
            "sys.argv", [sys.executable]
        ):
            self.assertIsNone(main())

        help_text, _usage, engine_help_text = err.getvalue().rpartition("usage")
        engine_help_text = _usage + engine_help_text
        self.assertEqual(
            engine_help_text,
            "usage: python -m ml_params [-h] [--version] [--engine {tensorflow}]\n"
            "python -m ml_params: error: --engine must be provided,"
            " and from installed ml-params-* options\n",
        )
        self.assertEqual(
            help_text,
            "usage: python -m ml_params [-h] [--version] [--engine {tensorflow}]\n\n"
            "Consistent CLI for every popular ML framework.\n\n"
            "optional arguments:\n"
            "  -h, --help            show this help message and exit\n"
            "  --version             show program's version number and exit\n"
            "  --engine {tensorflow}\n"
            '                        ML engine, e.g., "TensorFlow", "JAX", "pytorch"\n',
        )
        self.assertEqual(e.exception.code, SystemExit(2).code)

        self.assertEqual(out.getvalue(), "")

        # With engine set
        mod = "ml-params-tensorflow"
        if mod not in sys.modules:
            sys.modules[mod] = ml_params  # TODO: Some fake test module

        env_backup = dict(environ.items())
        environ["ML_PARAMS_ENGINE"] = "tensorflow"
        _argv = ["--engine", "tensorflow", "-h"]
        with self.assertRaises(SystemExit) as e, patch(
            "sys.stdout", new_callable=StringIO
        ) as out, patch("sys.stderr", new_callable=StringIO) as err, patch(
            "sys.argv", [sys.executable] + _argv
        ), patch(
            "os.environ", environ
        ):
            self.assertIsNone(main(_argv))
        self.assertEqual(e.exception.code, SystemExit(0).code)
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(
            out.getvalue(),
            "Adding: load_data_parser ;\n"
            "Adding: load_model_parser ;\n"
            "Adding: train_parser ;\n"
            "usage: python -m ml_params [-h] [--version] [--engine {tensorflow}]\n"
            "                           {load_data,load_model,train} ...\n\n"
            "Consistent CLI for every popular ML framework.\n\n"
            "positional arguments:\n"
            "  {load_data,load_model,train}\n"
            "                        subcommand to run. Hacked to be chainable.\n\n"
            "optional arguments:\n"
            "  -h, --help            show this help message and exit\n"
            "  --version             show program's version number and exit\n"
            "  --engine {tensorflow}\n"
            '                        ML engine, e.g., "TensorFlow", "JAX", "pytorch"\n',
        )

        del environ["ML_PARAMS_ENGINE"]
        environ.update(env_backup)

    def test_version(self) -> None:
        """ Tests that main will give you the right version """

        with self.assertRaises(SystemExit) as e, patch(
            "sys.stdout", new_callable=StringIO
        ) as out:
            self.assertIsNone(main(["--version"]))
        self.assertEqual(
            out.getvalue(), "python -m ml_params {}\n".format(ml_params.__version__)
        )
        self.assertEqual(e.exception.code, SystemExit(0).code)

    @skipIf(sys.version_info[:2] == (3, 5), "Enums are broken in 3.5?")
    def test__build_parser(self) -> None:
        """ Basic test for `_build_parser` """
        self.assertIsInstance(_build_parser(), ImportArgumentParser)

    def test_string_types(self) -> None:
        """ Test `string_types` values """
        with patch("sys.version", "3"):
            import ml_params.__main__

            self.assertTupleEqual(ml_params.__main__.string_types, (str,))
        with patch("sys.version", "2"):
            import ml_params.__main__

            self.assertTupleEqual(ml_params.__main__.string_types, (str,))

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
