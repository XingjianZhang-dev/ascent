# The ASCENT writer: structure, schemas, and scope

This document describes how the target-blind writer builds external state
(manuscript §4.2 and Table 1): what each evaluated task family supplies, what
all families share, one released row traced from input to prompt, and the
scope of the evidence (manuscript §7, "Schema scope"; Appendix D).

## 1. Structure

The writer has three parts.

- **Recogniser.** Scans the input that precedes the question and emits typed
  records. Every record carries its fields, its character position in the
  input, and the verbatim span it was read from.
- **State update.** Folds the records, in document order, into bounded state
  built from three primitives: a last-write map, an append-only log, and a
  counter. Each family composes these primitives in its own way.
- **Question map.** Recognises the question form and names the state keys from
  which the readout may select.

The record types, their composition into state, and the question forms are a
family's *schema*. Everything else is shared by all families:

| Shared component | Where it is implemented or recorded |
|---|---|
| Write boundary: nothing from the question onward is read while writing | every `read_*` function in `ascent/`; for BABILong, instrumented per row by `ascent/target_blindness.py` |
| Provenance: character position and verbatim span on every record | the record types `BabiFact`, `RulerFact`, `FrequencyEntry` |
| Registered slot bounds | `history_slots`, `memory_slots` in `configs/` |
| Nested budgets fixed before evaluation | `state_fact_slots_by_endpoint`, `state_budget_rule` in `configs/` |
| Fail-closed validation | `reports/C15_FAIL_CLOSED_VALIDATION_AUDIT_2026-09-20.md` |

## 2. The schemas of the evaluated families (manuscript Table 1)

| Family | Document | Recogniser → record | State (bound) | Question → keys | Budget s(m) | Code |
|---|---|---|---|---|---|---|
| BABILong QA1–QA3 | book text (PG-19) with interleaved event sentences | 4 patterns over closed person, place and object sets → move, pick-up, drop, transfer (person; place, object, recipient) | last-write maps person→place and object→holder, joined to object→place; per-object place log (256) | 3 forms → person; object; object and place | 1/2/3 fact slots; 4 at 7B–8B | `ascent/babilong_memory.py`: `extract_babilong_events`, `read_babilong` |
| BABILong QA7–QA8 | as above | same records | per-person inventory log; count of held objects | 2 forms → person | 1/2/8 fact slots | same module; `project_babilong_inventory` |
| RULER multi-query retrieval | essay text with key–value needles | 1 pattern → (key, value) | last-write key→value map (256) | keys named in the question | 40/69/91 replay tokens (Qwen2.5); 37/55/97 (SmolLM2) | `ascent/ruler_memory.py`: `read_ruler_niah` |
| RULER common-word aggregation | numbered list of words | 1 pattern → list item | exact counter (512 or 1,024; overflow aborts) | frequency rank | 3/7/10 words (Qwen2.5); 3/5/10 (SmolLM2) | `ascent/ruler_aggregation_memory.py`: `read_ruler_common_words` |

Budgets are listed in order of reader size. The configurations that register
every bound and budget are listed in `VERIFICATION.md` §2.

**Adapting the writer to a new document family** means supplying its schema:
a recogniser, a composition of the three primitives, and a question map. For
the two RULER families each recogniser is a single regular expression and each
state a single primitive. The rest of the pipeline sees a recogniser only
through its output, so any extractor, rule-based or learned, that emits
position-anchored verbatim records before the question reuses the same state
primitives, budget, and validation unchanged.

## 3. BABILong in detail

The BABILong recogniser (`extract_babilong_events`) recognises four event
types, each by a fixed regular expression over the closed bAbI sets of people,
locations and objects:

| Event kind | Pattern (paraphrased) | Fields |
|---|---|---|
| `move` | *\<person\> went (back)/travelled/moved/journeyed to the \<location\>.* | person, location |
| `pickup` | *\<person\> got/grabbed/took/picked up the \<object\> (there).* | person, object |
| `drop` | *\<person\> left/discarded/dropped/put down the \<object\> (there).* | person, object |
| `transfer` | *\<person\> gave/handed/passed the \<object\> to \<person\>.* | person, object, recipient |

Every event is recorded with its character position in the input and its
verbatim source sentence. The state update (`read_babilong`) maintains, in
stream order:

- **per-person location** (last `move` per person);
- **per-object location and location history**: an object's location is the
  location of whoever holds it at the moment it is dropped, or of the person
  after a transfer, so `pickup`/`drop`/`transfer` events are *assignment edges*
  that bind objects to people and people to places;
- **per-person inventory events** (only when QA7/QA8 are enabled for the panel)
  and **count summaries** (`project_babilong_inventory`: the objects a person
  currently carries, and their number as a count word).

The question map recognises five question forms (QA1 *Where is \<person\>?*,
QA2 *Where is the \<object\>?*, QA3 *Where was the \<object\> before the
\<location\>?*, QA7 *How many objects is \<person\> carrying?*, QA8 *What is
\<person\> carrying?*). The readout selects the supporting facts for the
queried entity from the state (for QA3, the two facts bracketing the queried
location in the object's history), keeps the last `s` of them (the endpoint's
fact-slot budget), and serialises them with one fixed canonical template per
event kind (`canonical_babilong_event_sources`):

```
<person> moved to the <location>.
<person> picked up the <object>.
<person> dropped the <object>.
<person> gave the <object> to <recipient>.
```

The reader then answers from the serialised facts and the question alone.

### Worked example (official 16K panel 1, row `00525ed1…`)

Input: 66,280 characters of PG-19 background text with 14 bAbI sentences
interleaved. Question: *Where is the milk?* (QA2). The recogniser finds the 14
events, e.g.

```
pos 2196   pickup   "Mary grabbed the milk there."       -> (mary, object=milk)
pos 4110   move     "John travelled to the bedroom."      -> (john, location=bedroom)
pos 13739  move     "Mary went back to the bathroom."     -> (mary, location=bathroom)
...
```

The state update tracks Mary's location and the milk's holder. The QA2
readout selects the milk's supporting facts (the last move of its holder and
the drop) and, with a 4-slot budget, retains:

```
mary moved to the hallway.
mary dropped the milk.
```

(canonicalised from *"Mary moved to the hallway."* and *"Mary discarded the
milk there."*). The persistent state for this row is 702 bytes; the ASCENT
prompt is 72 tokens against 15,689 for the Foundation prompt. The
target-blind parser's own reconstruction of the answer (*hallway*) is compared
with the reference target only *after* the prompt is built, as a
parser-accuracy audit; it never enters the state or the prompt.

### Inputs without bAbI-style event sentences

On such an input `extract_babilong_events` returns no records. For the QA1–QA3
question forms, `read_babilong` then has no state entry for the queried person
or object and raises, so the invocation aborts (fail-closed, manuscript §4.1)
instead of producing an answer from empty state. The BABILong recogniser
matches closed vocabularies and does not handle open-vocabulary entities,
paraphrase, or coreference; for non-templated text the recogniser is the
component to replace (§2), for example by an information-extraction or
LLM-based extractor writing the same typed records. Extending the writer to
non-templated natural text is the principal open direction named in
manuscript §7.

## 4. Target-blindness and scope are separate properties

| Property | Meaning | Evidence |
|---|---|---|
| **Target-blindness** | The writer and readout never see the answer. | Structural, and instrumented per row (`ascent/target_blindness.py`; `artifacts_revision/target_blindness_2026-09/`). |
| **Schema scope** | Which document structures the evidence covers. | The schemas in §2; manuscript §7 ("Schema scope") and the schema-blind ablation below. |

## 5. What the schema-blind ablation measures

To separate "structured external state helps frozen readers, and helps more
as they scale" from "this parser solves this benchmark", the revision runs the
identical pipeline (same panels, same per-endpoint slot budget `s(m)`, same
canonical serialiser, same frozen readers, same scorer) with the writer
replaced by generic, schema-free state constructions:

1. **sentence-window state**: the last `s` sentences of the document,
   order-preserving, a task-agnostic recency rule;
2. **lexical retrieval state**: the `s` sentences with the highest BM25 lexical
   overlap with the question, in document order (`ascent/generic_retrieval.py`);
3. **neural retrieval state**: BGE-M3 dense + sparse retrieval with the
   official cross-encoder reranker, top-`s` passages in document order (the
   retrieval stack of the baseline suite).

The pre-registered interpretation rule
(`artifacts_revision/schema_blind_2026-09/PREREGISTRATION.json`) fixed in
advance what each outcome means. None of the three generic writers satisfies
the registered co-scaling rule; at the same one-to-three-slot budget the
generic state is worse than no state at every scale, and the structured-fact
writer's advantage over each generic writer is 15 points at 0.5B and 89–99
points at 1.5B and 3B (manuscript Appendix D;
`artifacts_revision/schema_blind_2026-09/GENERIC_WRITER_ABLATION.json`). The
manuscript accordingly states the result for external state constructed
against the schemas in §2 (§7, "Schema scope").

## 6. Summary

- The writer is a recogniser, a state update over three primitives, and a
  question map; the schema is family-specific, and the write boundary,
  provenance, slot bounds, nested budgets and fail-closed validation are
  shared.
- Four evaluated families use it, two of them not bAbI-style (RULER
  multi-query retrieval and common-word aggregation), each with a
  single-pattern recogniser and a single-primitive state.
- The writer never sees the answer: structural, tested, and recorded on every
  BABILong row.
- In the schema-blind ablation, generic writers at the same budget fall below
  no state, and the structured-fact writer leads each of them by 15 to 99
  points; the evidence is stated for external state constructed against these
  schemas (manuscript §7).
