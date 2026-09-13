# VAWS knowledge corpus

Public Markdown knowledge and experience for vLLM-Ascend development. Git is the content
authority; OpenViking indexes and OVPack releases can be rebuilt.

Experience covers vLLM Ascend model development, operator integration, debugging,
performance and serving. Development of VAWS packages, control-plane services,
client wiring or the knowledge engine belongs with those projects.

The two stores are separate:

- `corpus/knowledge/` holds maintained conclusions. Use `knowledge_query` and
  `knowledge_explain`; check applicability against current code and evidence.
- `corpus/experience/` holds historical cases: the problem, investigation,
  action, observed outcome and remaining uncertainty. Use `experience_query`
  and `experience_explain`. Historical commands and implementations are clues
  for an investigation, not current operating instructions.

Both accept a Markdown title and non-empty body. Preserve conditions and limits,
including unsuccessful attempts and corrected explanations. Similar cases can
later inform maintained knowledge, skills or tools; storing a case does not
perform that conversion automatically.

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

Contribute only the redacted public copy prepared by the `vaws-knowledge`
package. Private source transcripts, paths, endpoints and credentials remain
local. The package preserves the selected kind when submitting through a fork.

Pull requests currently receive format and redaction checks, followed by
human review and merge.
Publishing a report does not establish that it has been reproduced elsewhere.

After a merge to `main`, the release workflow builds a dense OVPack on CPU
from the exact Git commit and publishes its manifest and pack together.
Configured clients download and verify the release, import its stored vectors,
then switch the shared version. Releases use schema `vaws-knowledge-release/2`
with `content.layout=kinds/v1`; older packs must be rebuilt. Failed updates
retain the previous version; project and candidate material stay local.

Runtime code and client setup belong to
[vaws-knowledge](https://github.com/vllm-ascend-workspace/vaws-knowledge).
This repository contains knowledge, publishing policy and thin CI entrypoints.

## References migrated from the development workspace

The shared corpus maintains the former workspace notes in
[`corpus/models/`](corpus/models/) (nine model observations),
[`corpus/debugging/`](corpus/debugging/) (twelve debugging observations), and
[`corpus/infra/`](corpus/infra/) (one SSH transport observation).
These directories are browsing aids, not required authoring categories.

Every migrated note links to its exact source commit and retains historical
conditions and missing evidence. Model titles explicitly mark their unverified
status; the conflicting GLM-5 layer counts remain unresolved. The old workspace
filenames remain stable here. Publication does not turn a historical observation
into a current configuration recommendation.

Maintain reusable vLLM, Ascend NPU, AI and inference infrastructure references
here. Tool commands, executable analyzer rules, API contracts and VAWS engineering
validation belong with their implementation. Workspace clients consume these
notes through the existing shared release; task agents need no migration step,
extra lookup requirement or local authoring mirror.

License: MIT.
