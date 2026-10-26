"""Hand-written parser contracts; no catalog labels or held-out queries are used."""

import json

import pytest

from catalog_search.attributes import FEATURE_NAMES, compare_attributes, extract_attributes


def status(query, title, kind, description=""):
    return [e["status"] for e in compare_attributes(query, title, description)["evidence"] if e["kind"] == kind]


def test_serializable_original_spans_and_stable_features():
    text = 'SSD 1 TB, output: 0.065 kW, width 2.54 cm; iPhone 11 Pro'
    attrs = extract_attributes(text)
    json.dumps(attrs)
    assert all(text[a["span"][0]:a["span"][1]] == a["text"] for a in attrs)
    result = compare_attributes("128 GB SSD", "256 GB SSD")
    assert tuple(result["features"]) == FEATURE_NAMES
    assert result["feature_vector"] == [result["features"][n] for n in FEATURE_NAMES]
    json.dumps(result)


@pytest.mark.parametrize("query,title,kind", [
    ("1 TB SSD", "1000 GB SSD", "storage"),
    ("1 TiB SSD", "1024 GiB SSD", "storage"),
    ("65 watts output", "output: 0.065 kW", "power"),
    ("width 1 inch", "width 25.4 mm", "length"),
    ("15.24 cm long cable", "6 inch long cable", "length"),
])
def test_unit_conversion(query, title, kind):
    assert status(query, title, kind) == ["match"]


def test_binary_decimal_uncertainty_not_forced_equality():
    assert status("1 TB SSD", "1024 GB SSD", "storage") == ["unknown"]
    assert status("8 GB RAM", "8 GiB RAM", "storage") == ["unknown"]


def test_english_thousands_separator():
    assert status("1,000 GB SSD", "1 TB SSD", "storage") == ["match"]
    assert extract_attributes("1,024 GB SSD")[0]["value"] == 1024e9


def test_unrelated_numerals_are_not_constraints():
    assert extract_attributes("book 1984 version 2 pack of 11") == []
    assert extract_attributes("4K USB3 2in1 3pack") == []


def test_storage_conflict_missing_and_ram_scope():
    assert status("128 GB phone", "256 GB phone", "storage") == ["conflict"]
    assert status("128 GB phone", "phone with large capacity", "storage") == ["unknown"]
    assert status("16 GB RAM", "16 GB storage", "storage") == ["unknown"]
    assert status("16 GB RAM", "16 GB RAM and 256 GB SSD", "storage") == ["match"]


def test_storage_speed_and_bits_are_not_byte_capacity():
    assert status("500 MB storage", "SSD transfers 500 MB/s", "storage") == ["unknown"]
    assert extract_attributes("500 MB/s SSD")[0]["unit"] == "byte/s"
    assert status("128 GB storage", "128 Gb storage", "storage") == ["unknown"]


@pytest.mark.parametrize("title", ["2 x 64 GB flash drives", "128 GB total, 2 x 64 GB drives", "64 GB / 128 GB variants"])
def test_bundle_and_variant_capacity_is_not_summed(title):
    assert status("128 GB flash drive", title, "storage") == ["unknown"]


def test_model_generation_and_suffix_are_distinct():
    assert status("iPhone 11 case", "case for iPhone 11 Pro", "model") == ["conflict"]
    assert status("iPhone 11 Pro", "iPhone 11 Pro Max", "model") == ["conflict"]
    assert status("iPhone 11", "iPhone 11", "model") == ["match"]
    assert status("iPhone 11", "Galaxy S21", "model") == ["unknown"]
    assert status("PS5 controller", "PlayStation 5 controller", "model") == ["match"]


def test_model_lists_and_sku():
    assert status("iPhone 12 case", "case for iPhone 11/12", "model") == ["unknown"]
    assert status("iPhone 11 case", "case for iPhone 11 and iPhone 12", "model") == ["match"]
    assert status("Logitech C920 webcam", "Logitech C920s webcam", "model") == ["conflict"]
    assert status("WH-1000XM4 headphones", "WH1000XM4 headphones", "model") == ["match"]


def test_query_alternative_is_unknown_not_two_required_values():
    assert status("128 GB or 256 GB SSD", "256 GB SSD", "storage") == ["unknown", "unknown"]


def test_power_nominal_difference_is_not_charger_incompatibility():
    assert status("65W charger", "100W USB PD charger", "power") == ["unknown"]
    assert status("65W charger", "65W charger", "power") == ["match"]
    assert status("exactly 65W charger", "100W charger", "power") == ["conflict"]
    assert status("65W charger", "input: 65W charger", "power") == ["unknown"]


@pytest.mark.parametrize("query,title,expected", [
    ("at least 65 W output", "100 W output", "match"),
    ("at least 65 W output", "45 W output", "conflict"),
    ("under 65 W", "65 W", "conflict"),
    ("over 65 W", "65-100 W", "unknown"),
    ("under 65 W", "45-65 W", "unknown"),
    ("up to 65 W", "65 W", "match"),
    ("13-15 inch laptop", "14 inch laptop", "match"),
    ("13-15 inch laptop", "16 inch laptop", "conflict"),
    ("14 inch laptop", "sleeve fits 13-15 inch laptop", "match"),
])
def test_bounds_ranges_and_supported_sizes(query, title, expected):
    kind = "power" if "W" in query else "length"
    assert status(query, title, kind) == [expected]


def test_ranges_with_distinct_units_and_descending_range():
    result = extract_attributes("width 1 inch to 3 cm")
    assert len(result) == 1
    assert result[0]["lower"] == pytest.approx(25.4)
    assert result[0]["upper"] == pytest.approx(30)
    assert status("width 3-1 cm", "width 2 cm", "length") == ["unknown"]
    assert status("between 13 and 15 inch laptop", "14 inch laptop", "length") == ["match"]
    assert status("13 and 15 inch laptop", "14 inch laptop", "length") == ["unknown"]
    assert status("not 13-15 inch laptop", "14 inch laptop", "length") == ["unknown"]


@pytest.mark.parametrize("query,title", [
    ("1/2 inch socket", "12.7 mm socket"),
    ("1 1/2 inch wide belt", "width 38.1 mm belt"),
    ("1 and 1/2 inch wide belt", "width 38.1 mm belt"),
    ("1+1/4 inch drain", "31.75 mm drain"),
    ("1½ inch drain", "38.1 mm drain"),
    ("¾ inch drain", "19.05 mm drain"),
])
def test_fractional_inches(query, title):
    assert status(query, title, "length") == ["match"]


def test_incomplete_and_invalid_fractions_abstain():
    assert status("1 +/4in drain", "4in drain", "length") == ["unknown"]
    assert extract_attributes("1/0 inch") == []
    assert status("00m bracket", "1m bracket", "length") == ["unknown"]


def test_negation_and_clause_boundary():
    assert status("not 128 GB", "128 GB SSD", "storage") == ["conflict"]
    assert status("not 128 GB", "256 GB SSD", "storage") == ["match"]
    assert status("iPhone 11 case", "not for iPhone 11", "model") == ["conflict"]
    attrs = extract_attributes("not 128 GB but 256 GB")
    assert [a["operator"] for a in attrs] == ["neq", "eq"]


def test_size_semantic_roles_and_dimension_ambiguity():
    assert status("width 10 cm", "height 10 cm", "length") == ["unknown"]
    assert status("width 20 cm", "10 x 20 cm frame", "length") == ["unknown"]
    assert status("13 inch laptop", "13.3 inch laptop", "length") == ["unknown"]
    assert not extract_attributes("2 in 1 convertible")
    assert status("10 m cable", "shoe size 10 M", "length") == ["unknown"]


def test_equivalent_wattage_does_not_match_consumption():
    assert status("60 W bulb", "8 W LED bulb, 60 W equivalent", "power") == ["unknown"]
    assert status("30w 48l folding table", "30W lamp", "power") == ["unknown"]
    assert status("80w x 60l tapestry", "80W heater", "power") == ["unknown"]


def test_brand_manufacturer_vs_compatibility():
    assert status("Samsung phone", "Sony phone", "brand") == ["conflict"]
    assert status("Samsung charger", "Anker charger for Samsung", "brand") == ["match"]
    assert status("Samsung phone", "case for Samsung phone", "brand") == ["unknown"]
    assert extract_attributes("apple slicer with apple fruit") == []
    assert status("Apple iPhone 11", "Apple iPhone 11", "brand") == ["match"]
    assert status("Lenovo or Dell laptop", "Lenovo laptop", "brand") == ["unknown", "unknown"]
    assert status("laptop with Microsoft Office", "Dell laptop", "brand") == ["unknown"]
    assert status("case without an Apple pencil", "Apple case", "brand") == ["unknown"]


def test_generic_numbers_and_technical_ratings_are_not_model_ids():
    assert not extract_attributes("binder4 MERV8 WiFi6 Bluetooth5 IP67 DDR4")
    attrs = extract_attributes("model binder4")
    assert attrs[0]["kind"] == "model" and attrs[0]["value"] == "binder4"


def test_description_can_support_but_not_assert_conflict_alone():
    assert status("128 GB SSD", "portable SSD", "storage", "Storage: 128 GB") == ["match"]
    assert status("128 GB SSD", "portable SSD", "storage", "Storage: 256 GB") == ["unknown"]
    assert status("128 GB SSD", "256 GB SSD", "storage", "Storage: 128 GB") == ["unknown"]


def test_empty_and_input_type_contract():
    result = compare_attributes("", "plain item")
    assert not result["evidence"]
    assert all(value == 0 for value in result["feature_vector"])
    with pytest.raises(TypeError):
        extract_attributes(None)
