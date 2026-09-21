from ascent.babilong_memory import (
    canonical_babilong_event_sources,
    irrelevant_babilong_facts,
    project_babilong_inventory,
    read_babilong,
    resolve_babilong_event_sources,
)


def test_canonical_event_sources_remove_deixis_without_location_injection() -> None:
    text = (
        "Mary went to the bathroom. Mary picked up the milk there. "
        "Mary journeyed to the office. Mary dropped the milk there."
    )
    rendered = list(canonical_babilong_event_sources(text).values())
    assert rendered == [
        "mary moved to the bathroom.",
        "mary picked up the milk.",
        "mary moved to the office.",
        "mary dropped the milk.",
    ]
    assert all(" there" not in source and " at the " not in source for source in rendered)


def test_qa1_returns_latest_person_location() -> None:
    text = "Mary went to the garden. filler Mary moved to the office."
    read = read_babilong(text, "Where is Mary?")
    assert read.task == "qa1"
    assert read.answer == "office"
    assert read.supporting_fact_count == 1


def test_qa2_tracks_object_with_its_owner() -> None:
    text = (
        "Mary went to the garden. Mary got the milk there. filler "
        "Mary travelled to the hallway."
    )
    read = read_babilong(text, "Where is the milk?")
    assert read.answer == "hallway"
    assert read.supporting_fact_count == 2
    assert [fact.kind for fact in read.facts] == ["pickup", "move"]


def test_qa2_drop_uses_current_person_location() -> None:
    text = (
        "John picked up the apple. John journeyed to the kitchen. "
        "John discarded the apple there."
    )
    read = read_babilong(text, "Where is the apple?")
    assert read.answer == "kitchen"
    assert [fact.kind for fact in read.facts] == ["move", "drop"]


def test_qa3_returns_location_before_named_location() -> None:
    text = (
        "Sandra went to the office. Sandra grabbed the football. "
        "Sandra moved to the bathroom."
    )
    read = read_babilong(text, "Where was the football before the bathroom?")
    assert read.answer == "office"
    assert read.supporting_fact_count == 3
    assert [fact.kind for fact in read.facts] == ["move", "pickup", "move"]


def test_resolved_event_sources_replace_there_from_stream_state_only() -> None:
    text = (
        "Sandra went to the office. Sandra grabbed the football there. "
        "Sandra moved to the bathroom. Sandra dropped the football there."
    )
    resolved = resolve_babilong_event_sources(text)
    assert list(resolved.values()) == [
        "sandra moved to the office.",
        "sandra picked up the football at the office.",
        "sandra moved to the bathroom.",
        "sandra dropped the football at the bathroom.",
    ]


def test_query_is_not_required_during_stream_writes() -> None:
    text = "Daniel moved to the bedroom. Daniel took the apple."
    qa1 = read_babilong(text, "Where is Daniel?")
    qa2 = read_babilong(text, "Where is the apple?")
    assert qa1.persistent_payload_bytes == qa2.persistent_payload_bytes
    assert qa1.answer == qa2.answer == "bedroom"


def test_registered_location_tasks_do_not_allocate_inventory_event_log() -> None:
    text = (
        "Daniel moved to the bedroom. Daniel took the apple. "
        "Daniel dropped the apple."
    )
    location_only_qa1 = read_babilong(
        text, "Where is Daniel?", enabled_tasks=("qa1", "qa2", "qa3")
    )
    location_only_qa2 = read_babilong(
        text, "Where is the apple?", enabled_tasks=("qa1", "qa2", "qa3")
    )
    all_tasks = read_babilong(text, "Where is Daniel?")
    assert (
        location_only_qa1.persistent_payload_bytes
        == location_only_qa2.persistent_payload_bytes
    )
    assert location_only_qa1.persistent_payload_bytes < all_tasks.persistent_payload_bytes


def test_question_task_must_be_enabled_before_stream_processing() -> None:
    try:
        read_babilong(
            "Mary went to the garden.",
            "Where is Mary?",
            enabled_tasks=("qa7", "qa8"),
        )
    except ValueError as error:
        assert "absent from enabled_tasks" in str(error)
    else:
        raise AssertionError("a disabled question task must be rejected")


def test_history_capacity_is_validated() -> None:
    try:
        read_babilong("Mary went to the garden.", "Where is Mary?", history_slots=1)
    except ValueError as error:
        assert "greater than one" in str(error)
    else:
        raise AssertionError("history_slots=1 must be rejected")


def test_irrelevant_control_excludes_support_people_and_objects() -> None:
    text = (
        "Mary went to the office. Mary took the apple. Mary moved to the garden. "
        "John journeyed to the hallway. Sandra grabbed the milk."
    )
    read = read_babilong(text, "Where is the apple?")
    irrelevant = irrelevant_babilong_facts(text, read, count=2)
    assert {fact.person for fact in irrelevant}.isdisjoint(
        {fact.person for fact in read.facts}
    )
    assert all(fact.object_name != "apple" for fact in irrelevant)


def test_qa7_counts_inventory_after_pickups_drops_and_transfers() -> None:
    text = (
        "Mary got the milk there. Mary grabbed the apple there. "
        "Mary gave the milk to Sandra. Sandra handed the milk to Mary. "
        "Mary dropped the apple there."
    )
    read = read_babilong(text, "How many objects is Mary carrying?")
    assert read.task == "qa7"
    assert read.answer == "one"
    assert [fact.kind for fact in read.facts] == [
        "pickup",
        "pickup",
        "transfer",
        "transfer",
        "drop",
    ]


def test_qa8_lists_current_objects_in_first_acquisition_order() -> None:
    text = (
        "Daniel got the milk there. Daniel took the football there. "
        "Daniel passed the milk to John. John gave the milk to Daniel."
    )
    read = read_babilong(text, "What is Daniel carrying?")
    assert read.task == "qa8"
    assert read.answer == "milk,football"
    assert read.supporting_fact_count == 4


def test_qa8_returns_nothing_after_final_transfer() -> None:
    text = "Sandra picked up the apple there. Sandra handed the apple to John."
    read = read_babilong(text, "What is Sandra carrying?")
    assert read.answer == "nothing"
    assert [fact.kind for fact in read.facts] == ["pickup", "transfer"]


def test_qa8_preserves_first_acquisition_order_after_reacquisition() -> None:
    text = (
        "Daniel got the football there. Daniel grabbed the apple there. "
        "Daniel dropped the football there. Daniel grabbed the football there."
    )
    read = read_babilong(text, "What is Daniel carrying?")
    assert read.answer == "football,apple"


def test_irrelevant_control_excludes_transfer_recipient() -> None:
    text = (
        "Mary took the apple. Mary gave the apple to Sandra. "
        "Sandra went to the office. John grabbed the milk."
    )
    read = read_babilong(text, "What is Sandra carrying?")
    irrelevant = irrelevant_babilong_facts(text, read, count=2)
    assert all(fact.person not in {"mary", "sandra"} for fact in irrelevant)


def test_inventory_projection_refines_a_retained_event_suffix() -> None:
    text = (
        "Mary got the milk there. Mary grabbed the apple there. "
        "Mary dropped the milk. Mary took the football there."
    )
    read = read_babilong(text, "What is Mary carrying?")
    one = project_babilong_inventory(read.facts[-1:], read.question)
    two = project_babilong_inventory(read.facts[-2:], read.question)
    full = project_babilong_inventory(read.facts, read.question)
    assert one.inventory_text == "football"
    assert two.inventory_text == "football"
    assert full.inventory_text == "apple,football"
    assert (one.count_word, two.count_word, full.count_word) == (
        "one",
        "one",
        "two",
    )
