"""Bounded causal memory for official BABILong event-state tasks.

The input stream is parsed before the question is inspected.  The reader
retains only a small world state (person locations, object ownership/current
locations, and bounded object-location histories), rather than the background
book text.  Retrieval therefore cannot use the reference answer.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass


_PEOPLE = r"Mary|John|Daniel|Sandra"
_LOCATIONS = r"bathroom|kitchen|bedroom|garden|hallway|office"
_OBJECTS = r"milk|apple|football"

_EVENT_PATTERNS = (
    (
        "move",
        re.compile(
            rf"\b(?P<person>{_PEOPLE}) "
            rf"(?:(?:went(?: back)?|travelled|moved|journeyed) to) the "
            rf"(?P<location>{_LOCATIONS})\.",
            re.IGNORECASE,
        ),
    ),
    (
        "pickup",
        re.compile(
            rf"\b(?P<person>{_PEOPLE}) "
            rf"(?:(?:got|grabbed|took) the|picked up the) "
            rf"(?P<object>{_OBJECTS})(?: there)?\.",
            re.IGNORECASE,
        ),
    ),
    (
        "drop",
        re.compile(
            rf"\b(?P<person>{_PEOPLE}) "
            rf"(?:(?:left|discarded|dropped) the|put down the) "
            rf"(?P<object>{_OBJECTS})(?: there)?\.",
            re.IGNORECASE,
        ),
    ),
    (
        "transfer",
        re.compile(
            rf"\b(?P<person>{_PEOPLE}) "
            rf"(?:(?:gave|handed|passed) the) "
            rf"(?P<object>{_OBJECTS}) to (?P<recipient>{_PEOPLE})\.",
            re.IGNORECASE,
        ),
    ),
)

_QA1 = re.compile(rf"Where is (?P<person>{_PEOPLE})\?\s*", re.IGNORECASE)
_QA2 = re.compile(rf"Where is the (?P<object>{_OBJECTS})\?\s*", re.IGNORECASE)
_QA3 = re.compile(
    rf"Where was the (?P<object>{_OBJECTS}) before the "
    rf"(?P<location>{_LOCATIONS})\?\s*",
    re.IGNORECASE,
)
_QA7 = re.compile(
    rf"How many objects is (?P<person>{_PEOPLE}) carrying\?\s*", re.IGNORECASE
)
_QA8 = re.compile(rf"What is (?P<person>{_PEOPLE}) carrying\?\s*", re.IGNORECASE)
_COUNT_WORDS = ("none", "one", "two", "three")


@dataclass(frozen=True)
class BabiFact:
    kind: str
    source: str
    character_position: int
    person: str
    location: str | None = None
    object_name: str | None = None
    recipient: str | None = None


@dataclass(frozen=True)
class BabiLongRead:
    task: str
    question: str
    facts: tuple[BabiFact, ...]
    answer: str
    persistent_payload_bytes: int

    @property
    def supporting_fact_count(self) -> int:
        return len(self.facts)


@dataclass(frozen=True)
class BabiInventoryProjection:
    person: str
    objects: tuple[str, ...]

    @property
    def count_word(self) -> str:
        return _COUNT_WORDS[len(self.objects)]

    @property
    def inventory_text(self) -> str:
        return ",".join(self.objects) if self.objects else "nothing"


@dataclass(frozen=True)
class _LocationState:
    location: str
    facts: tuple[BabiFact, ...]


def extract_babilong_events(input_text: str) -> tuple[BabiFact, ...]:
    """Return all recognized stream events without inspecting a question."""
    events: list[BabiFact] = []
    for kind, pattern in _EVENT_PATTERNS:
        for match in pattern.finditer(input_text):
            values = {
                key: value.lower()
                for key, value in match.groupdict().items()
                if value is not None
            }
            events.append(
                BabiFact(
                    kind=kind,
                    source=match.group(0),
                    character_position=match.start(),
                    person=values["person"],
                    location=values.get("location"),
                    object_name=values.get("object"),
                    recipient=values.get("recipient"),
                )
            )
    events.sort(key=lambda event: event.character_position)
    return tuple(events)


def resolve_babilong_event_sources(input_text: str) -> dict[int, str]:
    """Render stream events with causal locations and no question/target access.

    BABILong writes pickup and drop events with the anaphor ``there``.  Raw
    replay therefore asks a decoder to resolve a pronoun even though the
    bounded world state already had the person's location at write time.  This
    function exposes that causal state as an explicit event sentence.  The map
    is computed from the input stream alone before any question is inspected.
    """
    person_locations: dict[str, str] = {}
    resolved: dict[int, str] = {}
    for event in extract_babilong_events(input_text):
        if event.kind == "move":
            assert event.location is not None
            person_locations[event.person] = event.location
            source = f"{event.person} moved to the {event.location}."
        elif event.kind == "pickup":
            assert event.object_name is not None
            location = person_locations.get(event.person)
            suffix = f" at the {location}" if location is not None else ""
            source = f"{event.person} picked up the {event.object_name}{suffix}."
        elif event.kind == "drop":
            assert event.object_name is not None
            location = person_locations.get(event.person)
            suffix = f" at the {location}" if location is not None else ""
            source = f"{event.person} dropped the {event.object_name}{suffix}."
        else:
            assert event.kind == "transfer"
            assert event.object_name is not None and event.recipient is not None
            location = person_locations.get(event.recipient)
            suffix = f" at the {location}" if location is not None else ""
            source = (
                f"{event.person} gave the {event.object_name} to "
                f"{event.recipient}{suffix}."
            )
        resolved[event.character_position] = source
    return resolved


def canonical_babilong_event_sources(input_text: str) -> dict[int, str]:
    """Render deictic-free events without injecting a new location candidate.

    This target-blind diagnostic keeps the event semantics and chronology but
    removes lexical variation and the optional word ``there``. In contrast to
    :func:`resolve_babilong_event_sources`, it deliberately does not append a
    person's current location to pickup/drop/transfer events: the failed
    resolved-event diagnostic showed that this extra location becomes a strong
    distractor for QA3's temporal ``before`` readout.
    """
    canonical: dict[int, str] = {}
    for event in extract_babilong_events(input_text):
        if event.kind == "move":
            assert event.location is not None
            source = f"{event.person} moved to the {event.location}."
        elif event.kind == "pickup":
            assert event.object_name is not None
            source = f"{event.person} picked up the {event.object_name}."
        elif event.kind == "drop":
            assert event.object_name is not None
            source = f"{event.person} dropped the {event.object_name}."
        else:
            assert event.kind == "transfer"
            assert event.object_name is not None and event.recipient is not None
            source = (
                f"{event.person} gave the {event.object_name} to "
                f"{event.recipient}."
            )
        canonical[event.character_position] = source
    return canonical


def irrelevant_babilong_facts(
    input_text: str, read: BabiLongRead, *, count: int
) -> tuple[BabiFact, ...]:
    """Select recent facts whose people and objects are absent from the support.

    This is a target-blind negative control with the same fact-slot count as a
    relevant memory condition.  It excludes every supporting position, person,
    and object, so it cannot encode the selected query trajectory.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    support_positions = {fact.character_position for fact in read.facts}
    relevant_people = {fact.person for fact in read.facts}
    relevant_people.update(
        fact.recipient for fact in read.facts if fact.recipient is not None
    )
    relevant_objects = {
        fact.object_name for fact in read.facts if fact.object_name is not None
    }
    candidates = [
        fact
        for fact in extract_babilong_events(input_text)
        if fact.character_position not in support_positions
        and fact.person not in relevant_people
        and fact.object_name not in relevant_objects
    ]
    return tuple(candidates[-count:])


def _unique_chronological(facts: tuple[BabiFact, ...]) -> tuple[BabiFact, ...]:
    by_position = {fact.character_position: fact for fact in facts}
    return tuple(by_position[position] for position in sorted(by_position))


def project_babilong_inventory(
    facts: tuple[BabiFact, ...], question: str
) -> BabiInventoryProjection:
    """Replay a retained event suffix without reading a reference answer."""
    match = _QA7.fullmatch(question) or _QA8.fullmatch(question)
    if match is None:
        raise ValueError("inventory projection requires a QA7 or QA8 question")
    person = match.group("person").lower()
    present: set[str] = set()
    first_acquisition: dict[str, int] = {}
    for fact in sorted(facts, key=lambda value: value.character_position):
        if fact.object_name is None:
            continue
        object_name = fact.object_name
        add = (fact.kind == "pickup" and fact.person == person) or (
            fact.kind == "transfer" and fact.recipient == person
        )
        remove = (
            (fact.kind == "drop" and fact.person == person)
            or (fact.kind == "pickup" and fact.person != person)
            or (fact.kind == "transfer" and fact.person == person)
        )
        if remove:
            present.discard(object_name)
        if add:
            present.add(object_name)
            first_acquisition.setdefault(object_name, fact.character_position)
    objects = tuple(sorted(present, key=lambda value: first_acquisition[value]))
    return BabiInventoryProjection(person, objects)


def _payload_bytes(
    person_locations: dict[str, _LocationState],
    object_locations: dict[str, _LocationState],
    object_history: dict[str, deque[_LocationState]],
    inventory_events: dict[str, list[BabiFact]],
) -> int:
    strings: list[str] = []
    for person, state in person_locations.items():
        strings.extend((person, state.location, *(fact.source for fact in state.facts)))
    for object_name, state in object_locations.items():
        strings.extend(
            (object_name, state.location, *(fact.source for fact in state.facts))
        )
    for object_name, history in object_history.items():
        strings.append(object_name)
        for state in history:
            strings.extend((state.location, *(fact.source for fact in state.facts)))
    for person, events in inventory_events.items():
        strings.extend((person, *(event.source for event in events)))
    return sum(len(value.encode("utf-8")) for value in strings)


def read_babilong(
    input_text: str,
    question: str,
    *,
    history_slots: int = 256,
    enabled_tasks: tuple[str, ...] = ("qa1", "qa2", "qa3", "qa7", "qa8"),
) -> BabiLongRead:
    """Read an enabled BABILong task set without consulting the target answer.

    ``enabled_tasks`` is experiment-level state fixed before stream processing.
    It prevents state allocation for task families absent from a registered panel.
    """
    if history_slots <= 1:
        raise ValueError("history_slots must be greater than one")
    supported_tasks = {"qa1", "qa2", "qa3", "qa7", "qa8"}
    if not enabled_tasks or not set(enabled_tasks).issubset(supported_tasks):
        raise ValueError("enabled_tasks contains no tasks or unsupported tasks")
    inventory_enabled = bool(set(enabled_tasks) & {"qa7", "qa8"})

    person_locations: dict[str, _LocationState] = {}
    object_locations: dict[str, _LocationState] = {}
    object_owners: dict[str, str] = {}
    owner_objects: defaultdict[str, set[str]] = defaultdict(set)
    pickup_facts: dict[str, BabiFact] = {}
    first_acquisition_position: dict[tuple[str, str], int] = {}
    inventory_events: defaultdict[str, list[BabiFact]] = defaultdict(list)
    object_history: dict[str, deque[_LocationState]] = defaultdict(
        lambda: deque(maxlen=history_slots)
    )

    def append_history(object_name: str, state: _LocationState) -> None:
        history = object_history[object_name]
        if history and history[-1].location == state.location:
            # A pickup/drop at the same place is not a new location in the
            # object's trajectory.  Preserve the first-arrival evidence.
            return
        else:
            history.append(state)

    # All writes complete before the question is parsed.
    for event in extract_babilong_events(input_text):
        if event.kind == "move":
            assert event.location is not None
            person_state = _LocationState(event.location, (event,))
            person_locations[event.person] = person_state
            for object_name in tuple(owner_objects[event.person]):
                facts = _unique_chronological((pickup_facts[object_name], event))
                state = _LocationState(event.location, facts)
                object_locations[object_name] = state
                append_history(object_name, state)
            continue

        assert event.object_name is not None
        object_name = event.object_name
        if event.kind == "pickup":
            old_owner = object_owners.get(object_name)
            if old_owner is not None and old_owner != event.person:
                owner_objects[old_owner].discard(object_name)
                if inventory_enabled:
                    inventory_events[old_owner].append(event)
            object_owners[object_name] = event.person
            owner_objects[event.person].add(object_name)
            if inventory_enabled and old_owner != event.person:
                inventory_events[event.person].append(event)
                first_acquisition_position.setdefault(
                    (event.person, object_name), event.character_position
                )
            pickup_facts[object_name] = event
            person_state = person_locations.get(event.person)
            if person_state is not None:
                facts = _unique_chronological((*person_state.facts, event))
                state = _LocationState(person_state.location, facts)
                object_locations[object_name] = state
                append_history(object_name, state)
            continue

        if event.kind == "transfer":
            assert event.recipient is not None
            old_owner = object_owners.get(object_name)
            if old_owner is not None and old_owner != event.recipient:
                owner_objects[old_owner].discard(object_name)
                if inventory_enabled:
                    inventory_events[old_owner].append(event)
            object_owners[object_name] = event.recipient
            owner_objects[event.recipient].add(object_name)
            if inventory_enabled and old_owner != event.recipient:
                inventory_events[event.recipient].append(event)
                first_acquisition_position.setdefault(
                    (event.recipient, object_name), event.character_position
                )
            pickup_facts[object_name] = event
            recipient_state = person_locations.get(event.recipient)
            if recipient_state is not None:
                facts = _unique_chronological((*recipient_state.facts, event))
                state = _LocationState(recipient_state.location, facts)
                object_locations[object_name] = state
                append_history(object_name, state)
            continue

        old_owner = object_owners.get(object_name)
        if old_owner is not None:
            owner_objects[old_owner].discard(object_name)
            if inventory_enabled:
                inventory_events[old_owner].append(event)
        object_owners.pop(object_name, None)
        person_state = person_locations.get(event.person)
        if person_state is not None:
            # QA2 requires the location move and drop.  QA3 history also needs
            # the pickup to establish which object's path is being followed.
            current_facts = _unique_chronological((*person_state.facts, event))
            object_locations[object_name] = _LocationState(
                person_state.location, current_facts
            )
            history_facts = current_facts
            if object_name in pickup_facts:
                history_facts = _unique_chronological(
                    (pickup_facts[object_name], *current_facts)
                )
            append_history(
                object_name, _LocationState(person_state.location, history_facts)
            )

    payload = _payload_bytes(
        person_locations, object_locations, object_history, inventory_events
    )
    if match := _QA1.fullmatch(question):
        if "qa1" not in enabled_tasks:
            raise ValueError("question task qa1 is absent from enabled_tasks")
        state = person_locations[match.group("person").lower()]
        return BabiLongRead("qa1", question, state.facts, state.location, payload)

    if match := _QA2.fullmatch(question):
        if "qa2" not in enabled_tasks:
            raise ValueError("question task qa2 is absent from enabled_tasks")
        state = object_locations[match.group("object").lower()]
        return BabiLongRead("qa2", question, state.facts, state.location, payload)

    if match := _QA3.fullmatch(question):
        if "qa3" not in enabled_tasks:
            raise ValueError("question task qa3 is absent from enabled_tasks")
        object_name = match.group("object").lower()
        target_location = match.group("location").lower()
        history = tuple(object_history[object_name])
        target_index = max(
            index
            for index, state in enumerate(history)
            if index > 0 and state.location == target_location
        )
        previous = history[target_index - 1]
        target = history[target_index]
        facts = _unique_chronological((*previous.facts, *target.facts))
        return BabiLongRead("qa3", question, facts, previous.location, payload)

    if match := _QA7.fullmatch(question):
        if "qa7" not in enabled_tasks:
            raise ValueError("question task qa7 is absent from enabled_tasks")
        person = match.group("person").lower()
        objects = owner_objects[person]
        return BabiLongRead(
            "qa7",
            question,
            tuple(inventory_events[person]),
            _COUNT_WORDS[len(objects)],
            payload,
        )

    if match := _QA8.fullmatch(question):
        if "qa8" not in enabled_tasks:
            raise ValueError("question task qa8 is absent from enabled_tasks")
        person = match.group("person").lower()
        objects = sorted(
            owner_objects[person],
            key=lambda object_name: first_acquisition_position[(person, object_name)],
        )
        answer = ",".join(objects) if objects else "nothing"
        return BabiLongRead(
            "qa8",
            question,
            tuple(inventory_events[person]),
            answer,
            payload,
        )

    raise ValueError(f"unsupported BABILong question: {question!r}")
