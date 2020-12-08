"""
Tests for utils
"""
from unittest import TestCase

import ml_params.utils
from ml_params.tests.utils_for_tests import unittest_main
from ml_params.utils import camel_case, to_d


class TestUtils(TestCase):
    """
    Tests whether utils work
    """

    def test_camel_case(self) -> None:
        """
        Tests whether `camel_case` camelCases
        """
        self.assertEqual(camel_case("foo"), "foo")
        self.assertEqual(camel_case("foo", upper=True), "Foo")

        self.assertEqual(camel_case("can_haz"), "canHaz")
        self.assertEqual(camel_case("can_haz", upper=True), "CanHaz")

    def test_to_d(self) -> None:
        """
        Tests whether `to_d` creates the right dictionary
        """
        self.assertDictEqual(to_d({}), {})
        self.assertListEqual(
            *map(
                sorted,
                (
                    to_d(ml_params.utils).keys(),
                    (ml_params.utils.__all__ + ["getmembers"]),
                ),
            )
        )

    def test_to_numpy(self) -> None:
        """ Many variants of `to_numpy` to test """
        self.assertIsNone(ml_params.utils.to_numpy(None))
        self.assertIs(
            ml_params.utils.to_numpy(
                ml_params.utils, K=type("builtins", (object,), {"__name__": "builtins"})
            ),
            ml_params.utils,
        )
        self.assertEqual(
            ml_params.utils.to_numpy(
                type("_", tuple(), {"as_numpy": lambda: "as_numpy"})
            ),
            "as_numpy",
        )
        self.assertEqual(
            ml_params.utils.to_numpy(type("_", tuple(), {"numpy": lambda: "numpy"})),
            "numpy",
        )
        image_label = {"image": None, "label": None}
        self.assertDictEqual(ml_params.utils.to_numpy(image_label), image_label)
        with self.assertRaises(TypeError) as e:
            self.assertIsNone(ml_params.utils.to_numpy(type))
        self.assertEqual(
            e.exception.args,
            TypeError("Unable to convert <class 'type'> to numpy").args,
        )


unittest_main()
