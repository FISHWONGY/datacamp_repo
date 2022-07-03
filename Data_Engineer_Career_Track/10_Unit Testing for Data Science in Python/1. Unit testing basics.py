## Your first unit test using pytest
# https://github.com/gutfeeling/univariate-linear-regression
# Import the pytest package
import pytest

# Import the function convert_to_int()
from preprocessing_helpers import convert_to_int

# Complete the unit test name by adding a prefix
def test_on_string_with_one_comma():
  # Complete the assert statement
  assert convert_to_int("2,081") == 2081

## Spotting and fixing bugs

# !pytest test_convert_to_int.py
___________________________________

def convert_to_int(string_with_comma):
    # Fix this line so that it returns an int, not a str
    return int(string_with_comma.replace(",", ""))
