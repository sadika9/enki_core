import pytest
import enki_py


def test_sum_as_string():
    assert enki_py.sum_as_string(1, 1) == "2"
