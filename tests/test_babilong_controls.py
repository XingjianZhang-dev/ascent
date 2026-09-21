from ascent.babilong_controls import matched_raw_event_fifo


def test_matched_raw_fifo_respects_exact_utf8_budget() -> None:
    text = (
        "Mary went to the garden. John moved to the office. "
        "Mary travelled to the kitchen."
    )
    budget = len("John moved to the office.Mary travelled to the kitchen.".encode())
    result = matched_raw_event_fifo(
        text, "Where is Mary?", count=1, budget_bytes=budget
    )
    assert result.payload_bytes <= budget
    assert result.cached_events == (
        "John moved to the office.",
        "Mary travelled to the kitchen.",
    )
    assert result.passages == ("Mary travelled to the kitchen.",)


def test_matched_raw_write_is_query_blind() -> None:
    text = "Mary went to the garden. John moved to the office."
    first = matched_raw_event_fifo(
        text, "Where is Mary?", count=1, budget_bytes=1000
    )
    second = matched_raw_event_fifo(
        text, "Where is John?", count=1, budget_bytes=1000
    )
    assert first.cached_events == second.cached_events
    assert first.payload_bytes == second.payload_bytes


def test_matched_raw_skips_event_larger_than_budget() -> None:
    result = matched_raw_event_fifo(
        "Mary went to the garden.",
        "Where is Mary?",
        count=1,
        budget_bytes=4,
    )
    assert result.cached_events == ()
    assert result.passages == ()
    assert result.payload_bytes == 0
