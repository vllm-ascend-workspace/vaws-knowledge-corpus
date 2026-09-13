# Knowledge and experience content repository

Keep maintained knowledge in `corpus/knowledge/` and historical cases in
`corpus/experience/`. Each document needs a title and non-empty body. Keep
source conditions when known; do not invent hardware, version, accuracy or
performance evidence. Historical observations and implementations remain dated
experience, not current guidance. Correct mistaken explanations while preserving
the observed outcome and uncertainty.

Preserve enough technical detail for an unfamiliar reader to understand why the
investigation or change mattered. For complex cases this can include the actual
call path, input layout, state or event lifetime, reference calculation and
discriminating results. Read the underlying source or output before adding a
claim; summaries alone are not new evidence. Keep simple cases concise. There is
no required article length or section template, and unresolved findings may
remain explicitly unresolved.

Experience here concerns vLLM Ascend models, operators and business execution.
Do not contribute VAWS package, control-plane, client wiring or knowledge-engine
development cases; keep those with their owning projects.

Knowledge uses fixed semantic category/entry paths, for example
`corpus/knowledge/graph/buffers.md`; identical text does not merge different
entries. Experiences retain stable case identities. Preserve existing filenames,
including old hash prefixes, when correcting titles, evidence or conclusions.
An experience's similar title or symptom does not prove it is the same case.
The package keeps the current content digest separately for change detection
and integrity. Agents can revise a local candidate by `ref`, select a knowledge
entry by `public_relpath="knowledge/CATEGORY/ENTRY.md"`, or correct an existing
shared case with `public_relpath="experience/CASE.md"`.

Optional `experience_feedback(ref, vote)` records `+1` when a published case
helped or `-1` when it misled the work. No reason or extra summary is required.
The package records each usage feedback as a minimal comment on the experience's
GitHub feedback Issue. Both positive and negative feedback accumulate, including
repeated feedback from the same account; neither removes earlier events.
Normal calls need only ref and vote. Reuse a failed call's returned `request_id`
only to retry that event; omit it for a new usage event. Feedback stays on GitHub,
outside the article and release pack. It does not certify correctness, change
retrieval ranking or automatically promote experience to knowledge.

Public contributions must pass the installed `vaws-knowledge` redaction and
Markdown checks. Do not commit private endpoints, user paths or credentials.

Runtime implementation belongs in `vllm-ascend-workspace/vaws-knowledge`.
CI uses a fixed reviewed package revision. It never imports Python modules
from a proposed corpus checkout. Human reviewers decide whether to merge.
