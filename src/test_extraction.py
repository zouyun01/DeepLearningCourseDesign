from answer_extraction import (
    extract_answer_text,
    extract_answer_value,
    extract_gold_answer_value,
    is_correct_completion,
    numeric_equal,
)


def test_final_answer_marker():
    x = "We compute 3+5=8. Final Answer: 8"
    assert extract_answer_text(x, strict=True) == "8"
    assert extract_answer_value(x, strict=True) == 8.0


def test_answer_is_marker():
    x = "After adding April and May, the answer is 72."
    assert extract_answer_text(x, strict=True) == "72"
    assert extract_answer_value(x, strict=True) == 72.0


def test_answer_equals_marker():
    x = "The total number of clips is answer = 72."
    assert extract_answer_text(x, strict=True) == "72"
    assert extract_answer_value(x, strict=True) == 72.0


def test_trailing_equals_marker():
    x = "April clips are 48 and May clips are 24, so total clips = 72."
    assert extract_answer_text(x, strict=True) == "72"


def test_intermediate_equals_not_strict_marker():
    x = "April clips are 48 and May clips are 48/2=24. The total is not stated."
    assert extract_answer_text(x, strict=True) is None


def test_hash_marker_only_for_gold():
    assert extract_answer_text("Reasoning... #### 8", strict=True) is None
    assert extract_gold_answer_value("Reasoning... #### 8") == 8.0


def test_gsm8k_gold():
    gold = "He has 3 + 5 = <<3+5=8>>8 apples. #### 8"
    assert extract_gold_answer_value(gold) == 8.0


def test_fraction():
    assert numeric_equal("1/2", "0.5")
    assert is_correct_completion("Reasoning... Final Answer: 1/2", "#### 0.5", strict=True)


def test_negative_decimal_commas():
    assert numeric_equal("-1,234.50", "-1234.5")


def test_lenient_last_number():
    x = "First I thought 5. Then recomputed and got 7."
    assert extract_answer_text(x, strict=True) is None
    assert extract_answer_value(x, strict=False) == 7.0


def test_incorrect():
    assert not is_correct_completion("Final Answer: 9", "#### 8", strict=True)


if __name__ == "__main__":
    test_final_answer_marker()
    test_answer_is_marker()
    test_answer_equals_marker()
    test_trailing_equals_marker()
    test_intermediate_equals_not_strict_marker()
    test_hash_marker_only_for_gold()
    test_gsm8k_gold()
    test_fraction()
    test_negative_decimal_commas()
    test_lenient_last_number()
    test_incorrect()
    print("All answer extraction tests passed.")
