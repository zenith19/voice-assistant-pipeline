import pytest

from app.pipeline import tools


@pytest.mark.parametrize("expression, expected", [
    ("23 * 17", 391),
    ("12 * (3 + 4)", 84),
    ("-5 + 2", -3),
    ("2 ** 10", 1024),
    ("7 % 3", 1),
    ("1 / 4", 0.25),
])
def test_calculate_valid(expression, expected):
    assert tools.calculate(expression)["result"] == expected


@pytest.mark.parametrize("expression", [
    "__import__('os').system('true')",  # no names / calls
    "abs(-1)",
    "'a' * 3",                           # no strings
    "1 / 0",
    "2 +",                               # syntax error
    "9 ** 9 ** 9",                       # unbounded exponent used to hang the server
    "True + 1",                          # bools are not numbers here
])
def test_calculate_rejects_unsafe_or_invalid(expression):
    result = tools.calculate(expression)
    assert "error" in result and "result" not in result


def test_dispatch_matches_schemas():
    schema_names = {t["function"]["name"] for t in tools.TOOLS}
    assert schema_names == set(tools.DISPATCH)


def test_weather_is_case_insensitive_and_has_fallback():
    assert tools.get_weather("  BERLIN ")["forecast"] == "21C and sunny"
    assert "forecast" in tools.get_weather("Atlantis")
