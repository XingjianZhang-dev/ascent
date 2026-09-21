from ascent.ruler_vt_memory import read_ruler_vt


def test_vt_read_selects_composed_chain_in_stream_order() -> None:
    text = """Memorize and track the chain(s) of variable assignment.
VAR AAA = 12345
VAR ZZZ = 99999
VAR BBB = VAR AAA
VAR CCC = VAR BBB
Question: Find all variables that are assigned the value 12345 in the text above."""
    read = read_ruler_vt(text)
    assert read.values == ("AAA", "BBB", "CCC")
    assert [fact.source for fact in read.facts] == [
        "VAR AAA = 12345",
        "VAR BBB = VAR AAA",
        "VAR CCC = VAR BBB",
    ]


def test_vt_read_ignores_solved_demonstration() -> None:
    text = """Memorize and track the chain(s) of variable assignment.
VAR OLD = 11111
Question: Find all variables that are assigned the value 11111 in the text above. Answer: OLD

Memorize and track the chain(s) of variable assignment.
VAR NEW = 22222
VAR NEXT = VAR NEW
Question: Find all variables that are assigned the value 22222 in the text above."""
    assert read_ruler_vt(text).values == ("NEW", "NEXT")


def test_vt_read_obeys_bounded_lru_capacity() -> None:
    text = """Memorize and track the chain(s) of variable assignment.
VAR AAA = 12345
VAR BBB = VAR AAA
Question: Find all variables that are assigned the value 12345 in the text above."""
    assert read_ruler_vt(text, memory_slots=1).values == ()
