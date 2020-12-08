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


unittest_main()
