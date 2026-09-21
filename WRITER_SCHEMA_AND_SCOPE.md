# The BABILong writer: schema, scope, and what it does and does not show

This document answers one question directly: **is the ASCENT writer used in
the BABILong experiments a structured-fact schema tailored to BABILong, or a
general-purpose external-memory writer?** It is the former. Anyone reading
`ascent/babilong_memory.py` (≈480 lines) will see this within minutes, so the
scope is stated here rather than left to be discovered.

Two properties must be kept apart throughout:

| Property | Meaning | Status |
|---|---|---|
| **Target-blindness** | The writer and readout never see the answer. | Provable and now instrumented per row (`ascent/target_blindness.py`; `artifacts_revision/target_blindness_2026-09/`). |
| **Task-generality** | The schema captures structures beyond the evaluated benchmark families. | **Not** established by target-blindness. It is a limitation of the present evidence; the schema-blind ablation in `artifacts_revision/schema_blind_2026-09/` measures how much of the gain survives without it. |

Target-blindness says nothing about generality, and generality says nothing
about target-blindness. Reviewer 2 asked about the second; the manuscript's
earlier text answered the first.

## 1. What the writer extracts

The writer is a deterministic parser over the input stream (`extract_babilong_events`)
followed by a small causal state machine (`read_babilong`). It recognises
exactly four event types, each by a fixed regular expression over a closed
vocabulary of people, locations and objects (the bAbI entity sets):

| Event kind | Pattern (paraphrased) | Fields |
|---|---|---|
| `move` | *\<person\> went (back)/travelled/moved/journeyed to the \<location\>.* | person, location |
| `pickup` | *\<person\> got/grabbed/took/picked up the \<object\> (there).* | person, object |
| `drop` | *\<person\> left/discarded/dropped/put down the \<object\> (there).* | person, object |
| `transfer` | *\<person\> gave/handed/passed the \<object\> to \<person\>.* | person, object, recipient |

Every event is recorded with its character position in the input and its
verbatim source sentence. From the event sequence the state machine maintains,
in stream order:

- **per-person location** (last `move` per person);
- **per-object location and location history** — an object's location is the
  location of whoever holds it at the moment it is dropped, or of the person
  after a transfer, so `pickup`/`drop`/`transfer` events are *assignment edges*
  that bind objects to people and people to places;
- **per-person inventory events** (only when QA7/QA8 are enabled for the panel)
  and **count summaries** (`project_babilong_inventory`: the objects a person
  currently carries, and their number as a count word).

The **readout** is question-conditioned and equally narrow: five regular
expressions recognise the five question forms (QA1 *Where is \<person\>?*,
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
interleaved. Question: *Where is the milk?* (QA2). The parser finds the 14
events, e.g.

```
pos 2196   pickup   "Mary grabbed the milk there."       -> (mary, object=milk)
pos 4110   move     "John travelled to the bedroom."      -> (john, location=bedroom)
pos 13739  move     "Mary went back to the bathroom."     -> (mary, location=bathroom)
...
```

The state machine tracks Mary's location and the milk's holder. The QA2
readout selects the milk's supporting facts — the last move of its holder and
the drop — and, with a 4-slot budget, retains:

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

## 2. What the schema was designed for

The four event kinds and five question forms are exactly the structures of
BABILong QA1–QA3 and QA7–QA8 (single/two/three supporting facts; counting;
lists). The RULER writers in `ascent/ruler_*_memory.py` are likewise
task-specific: word-frequency aggregation for CWE/FWE, key–value binding for
NIAH, and QA-passage retention for the RULER QA tasks. Each writer is a
structured-fact schema for a family of templated tasks. None of them parses
open-domain natural language.

Consequently the evidence in the paper establishes scale complementarity
**for external state constructed against such structures**: the claim is
scoped to structured-fact long-context tasks, and the writer is best
described as a benchmark-family parser feeding a bounded, target-blind,
query-conditioned memory.

## 3. What is expected on non-templated natural text

Applied unmodified to a document that does not contain bAbI-style sentences,
`extract_babilong_events` returns no events, the state is empty, and ASCENT
degrades to the Foundation reader with an empty memory block. The writer has
no mechanism for open-vocabulary entities, paraphrase, coreference beyond the
closed name set, or relations other than the four event kinds. Extending the
writer to non-templated text (an LLM- or IE-based event extractor writing
into the same typed state) is the principal open direction and is not
evaluated in this work.

## 4. What the schema-blind ablation measures

To separate "structured external state helps frozen readers, and helps more
as they scale" from "this parser solves this benchmark", the revision runs the
identical pipeline — same panels, same per-endpoint slot budget `s(m)`, same
canonical serialiser, same frozen readers, same scorer — with the writer
replaced by generic, schema-free state constructions:

1. **sentence-window state**: the last `s` sentences of the document,
   order-preserving, a task-agnostic recency rule;
2. **lexical retrieval state**: the `s` sentences with the highest BM25 lexical
   overlap with the question, in document order (`ascent/generic_retrieval.py`);
3. **neural retrieval state**: BGE-M3 dense + sparse retrieval with the
   official cross-encoder reranker, top-`s` passages in document order (the
   retrieval stack of the baseline suite).

The pre-registered interpretation rule
(`artifacts_revision/schema_blind_2026-09/PREREGISTRATION.json`) fixes in
advance what each outcome means: if the co-scaling pattern (increasing gain
with reader scale, positive adjacent increments) survives under a generic
writer, the pattern does not depend on the hand-built schema; if it collapses,
the schema is doing the work and the manuscript's claim is narrowed to
structured-fact-retrieval tasks. Either outcome is reported as found.

## 5. Summary for a reader in a hurry

- The writer never sees the answer: structural, tested, and recorded on every
  row.
- The writer is a BABILong/RULER-family parser: four event kinds, five
  question forms, closed entity vocabularies.
- The paper's claim is therefore scoped to structured-fact long-context tasks;
  generalisation to natural text is not demonstrated.
- The schema-blind ablation quantifies how much of the effect survives without
  the schema; the manuscript reports its outcome and narrows or keeps the
  claim accordingly.
