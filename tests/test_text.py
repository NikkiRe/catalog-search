"""Split/text invariants that preserve physical constraints without assigning labels."""

import pytest

from catalog_search.text import clean, document, family, stable_hash, tokens


def test_clean_handles_html_entities_unicode_and_missing_text():
    assert clean(None) == ""
    assert clean("<b>６５ W</b>&nbsp; USB &amp; cable\n") == "65 W USB & cable"
    assert tokens("６５W USB-C2.0 charger") == ["65w", "usb-c2.0", "charger"]


def test_clean_does_not_erase_literal_numeric_inequalities_as_html():
    assert clean("power < 10 W and > 5 W") == "power < 10 W and > 5 W"
    assert clean("size <5 and >2 inches") == "size <5 and >2 inches"


def test_clean_preserves_values_of_mixed_unicode_fractions():
    assert clean("pipe 1½ inch") == "pipe 1 1/2 inch"
    assert clean("¼ inch washer") == "1/4 inch washer"


@pytest.mark.parametrize("left,right", [
    ("  ６５Ｗ USB-C charger  ", "45 W usb-c charger"),
    ("12.5-inch shelf", "9,25 inch shelf"),
    ("2 x 4 inch frame", "8 x 10 inch frame"),
    ("iPhone14 case", "iphone15 case"),
    ("shelf 30cm wide", "shelf 50 cm wide"),
])
def test_numeric_templates_group_case_spacing_and_numeric_variants(left, right):
    assert family(left) == family(right)


@pytest.mark.parametrize("left,right", [
    ("3 1/2 inch pipe", "3.5 inch pipe"),
    ("3/4 inch pipe", "0.75 inch pipe"),
    ("1½ inch pipe", "1.5 inch pipe"),
])
def test_fractional_and_decimal_literals_do_not_leak_across_families(left, right):
    assert family(left) == family(right)


def test_family_preserves_nonnumeric_meaning_and_is_not_a_unit_conversion_oracle():
    assert family("red 5 inch shelf") != family("blue 5 inch shelf")
    assert family("pixel 8 case") != family("iphone 14 case")
    assert family("charger without cable") != family("charger with cable")
    assert family("chair 50 cm") != family("chair 20 inch")
    # These families prevent simple textual leakage; they do not prove semantic equivalence.


def test_seeded_hash_is_label_blind_order_independent_and_seed_sensitive():
    keys = ["query|product-c", "query|product-a", "query|product-b"]
    first = {key: stable_hash(key) for key in keys}
    assert first == {key: stable_hash(key) for key in reversed(keys)}
    assert all(0 <= value < 2**64 for value in first.values())
    assert stable_hash(keys[0], seed=1) != stable_hash(keys[0], seed=2)


def test_document_preserves_both_attribute_and_description_source_information():
    title, brand, text = document("<b>Chair</b>", None, "fabric blue", "width:18 inches")
    assert title == "Chair" and brand == ""
    assert text == "width:18 inches fabric blue"
