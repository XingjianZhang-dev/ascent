# ASCENT design risks and fail-closed resolutions

## R1 — a weak “matched raw” baseline can manufacture a win

On a single fresh key/value event, a raw cache that stores the exact key and
value can recover perfectly. Conversely, a baseline given the same noisy bits
and the exact registered likelihood must tie the ASCENT certified decoder.
Therefore it would be invalid to claim superiority by comparing the Bayes
decoder to an intentionally weak majority-vote decoder.

Resolution for the endpoint program:

- Report the same-signal exact decoder as a mandatory equality control.
- Evaluate exact raw key/value storage under identical total bytes on a stream
  with more events than it can retain; count key, value, timestamp, version,
  validity, addressing, and slot metadata for every method.
- Freeze event count, query distribution, overwrite rate, key entropy, and
  answer entropy across scales.
- Attribute any advantage separately to representation/compression, retrieval,
  decoding, and foundation interaction.
- Refuse Stage-B promotion if ASCENT does not beat exact raw storage on the
  preregistered capacity-pressure tasks.

## R2 — certified co-scaling can be entirely memory-driven

Increasing refinement rounds makes the certified gain grow even if the
foundation is unused. This validates the information theorem but does not show
that a larger foundation extracts more value.

Resolution: run a full two-factor design with foundation size and memory signal
varied independently. The confirmatory foundation-complementarity claim requires
a positive interaction interval after controlling for state bytes and decoder
capacity. The certified curve alone cannot satisfy this gate.

## R3 — raw foundation candidate NLL is not the theorem baseline

Public language models have arbitrary preferences among abstract answer tokens.
Using their raw candidate distribution would make the no-memory loss depend on
model scale and tokenizer artifacts.

Resolution: the certified benchmark uses the exact uniform permutation-
symmetrized head (`log M` NLL). Raw vocabulary/candidate NLL is reported only as
a leakage and external-validity diagnostic.

## R4 — checkpoint transfer and environment drift

The GPU node cannot directly access Hugging Face. Silent checkpoint or library
drift would make cross-scale comparisons unreliable.

Resolution: download immutable revisions locally, hash every weight/config/
tokenizer file, transfer to the data disk, re-hash remotely, and record the
source commit plus Torch/Transformers/CUDA/GPU versions in every result.

## R5 — exact raw KV can dominate the certified channel on simple recall

The formal task says that the write presents `(K,V)` before the query. A strong
raw baseline can therefore store a clean value plus an optimized key
fingerprint. On single-key associative recall, adding artificial BSC noise to
ASCENT's value code cannot create an information advantage over that baseline;
under the same bytes, compressed raw fingerprints with clean values may strictly
dominate. Comparing only against an uncompressed 128-bit-key cache would not be
reviewer-proof because the raw method can use the same registered fingerprint.

Resolution:

- Treat same-byte exact KV with an optimized fingerprint-length sweep as a
  mandatory strong baseline, not merely full-key raw storage.
- Do not require ASCENT to beat this baseline on the theorem unit test; the
  theorem establishes monotone added information, not superiority to an oracle
  that observes clean `(K,V)`.
- Seek practical advantage on finite-capacity overwrite, multi-event
  composition, semantic/latent writes, and natural text, where representation,
  retention, and foundation interaction matter.
- If the proposal's Stage-B wording is interpreted as requiring ASCENT to beat
  optimized exact KV on simple fresh recall, mark that criterion internally
  inconsistent and revise it before endpoint confirmation rather than weakening
  the baseline after seeing results.

## R6 — endpoint-specific noise shapes can fake a fixed-state scale effect

The original diagnostic sampled a noise tensor whose number of refinement
rounds equaled the endpoint's scale allocation. NumPy's random-number layout
then changed the first-round signal when the requested tensor shape changed.
Consequently, the nominally fixed one-round arm at 410M and 2.8B used different
realizations, so their observed difference could not be attributed to scale.

Resolution:

- Generate the maximum registered number of rounds exactly once from the
  common data seed, independent of the selected endpoint.
- Give every fixed and scaled arm a strict prefix of that common tensor.
- Record hashes of the labels, full signal, and each arm's prefix in every
  endpoint result; require matching hashes before a cross-endpoint contrast.
- Discard all endpoint results produced before this correction and rerun them
  from a clean committed tree.
