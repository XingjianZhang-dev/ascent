import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def target_subscripts(path: Path) -> list[int]:
    source = path.read_text()
    tree = ast.parse(source)
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "row":
            continue
        key = node.slice
        if isinstance(key, ast.Constant) and key.value == "target":
            lines.append(node.lineno)
    return sorted(lines)


def test_reference_target_is_used_only_for_fail_closed_audit_and_scoring() -> None:
    runner = ROOT / "experiments/run_babilong_prompt.py"
    source_lines = runner.read_text().splitlines()
    occurrences = target_subscripts(runner)
    assert len(occurrences) == 4
    snippets = [source_lines[line - 1].strip() for line in occurrences]
    assert snippets == [
        'if read.answer != row["target"]:',
        'foundation_score = float(foundation_answer == row["target"])',
        'ascent_score = float(ascent_answer == row["target"])',
        '"target": row["target"],',
    ]


def test_causal_memory_implementation_has_no_reference_target_access() -> None:
    memory_source = (ROOT / "ascent/babilong_memory.py").read_text()
    assert '"target"' not in memory_source
    assert "['target']" not in memory_source
    assert '["target"]' not in memory_source
