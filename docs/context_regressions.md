# Saved-card context regression gate

Run `python -m scripts.compare_context_cards BASELINE.json CANDIDATE.json EXPECTATIONS.json`
after replaying the **same saved click and frozen research** through a candidate
context reviewer or ordering path. Both files use `{"payload": ..., "result": ...}`.
The read-only command prints JSON and exits nonzero on failure.

Use zero-based `result.rows` indexes and IDs from the corresponding
`result.decomposed.verdict.assertions[].relevant[]`:

```json
{
  "must_retain": [{"row": 0, "assertion_id": "assertion-1", "unit_id": "source-1:p1:u1"}],
  "must_drop": [{"row": 0, "assertion_id": "assertion-1", "unit_id": "source-1:p2:u1"}]
}
```

Name at least one useful pair. An empty `must_drop` list is valid for a preservation
run. Review expectations before candidate inference; do not tune them to one example.
Unknown, duplicate, malformed or contradictory IDs fail.

The gate freezes payload, original claims, source units, research, counts, evidence,
statuses and summary. Context may reorder or be withheld with an unrelated secondary
reading, recorded in `decomposed.judgments` or `withheld_context`. Derived prose and
ordering audit may change. Passing is structural preservation plus explicit
expectations, not accuracy or release certification; no model, retrieval, browser
or network check runs here.
