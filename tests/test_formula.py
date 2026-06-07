import numpy as np
import pytest

from fastiso import counts_matrix, parse_formula


def test_parse_formula_plain_counts():
    assert parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}


def test_parse_formula_parentheses():
    assert parse_formula("(CH3)2O") == {"C": 2, "H": 6, "O": 1}


def test_parse_formula_adjacent_bracketed_groups():
    assert parse_formula("(CH3OH)2(HCl)2") == {
        "C": 2,
        "H": 10,
        "O": 2,
        "Cl": 2,
    }


def test_parse_formula_multiple_bracket_styles_and_nested_groups():
    assert parse_formula("K4[Fe(CN)6]") == {
        "K": 4,
        "Fe": 1,
        "C": 6,
        "N": 6,
    }
    assert parse_formula("{CH3[CH2]2}2O") == {"C": 6, "H": 14, "O": 1}


def test_parse_formula_rejects_mismatched_brackets():
    with pytest.raises(ValueError, match="expected closing bracket"):
        parse_formula("K4[Fe(CN]6)")


def test_counts_matrix_rejects_unknown_element():
    with pytest.raises(ValueError, match="not available"):
        counts_matrix(["C6H12O6Na"], ("C", "H", "O"))


def test_counts_matrix_uses_requested_element_order():
    counts = counts_matrix(["C6H12O6"], ("H", "C", "O"))

    np.testing.assert_array_equal(counts, np.array([[12, 6, 6]]))
