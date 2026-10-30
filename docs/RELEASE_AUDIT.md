# v0.1.0 release checklist audit — October 29, 2026

**Audit complete.** Phase 6 and the release were open at this October 29
snapshot; subsequent M6 closure is recorded below. Of the ten unchecked
items in the original Definition of Done, seven already have implementation or
documentation evidence, one (`cinder info`) is explicitly deferred beyond
v0.1.0, and two remain open: the final README review and the GitHub release.

This audit starts at `ecb2ff1` (M5 evaluation/closure). Runtime, headers, tests,
exporter/example sources, CMake, and the CI workflow are unchanged from hosted
revision `cd461c696aea932e6ea53e4093008d6463c42fa9`. The [M5 verification report](M5_VERIFICATION.md)
and its saved logs were then the latest full-suite evidence: local Debug 46/46,
Release and ASan/UBSan 37/37 each, four zero-leak native checks, and five passing
hosted jobs. This documentation audit does not claim a new full-suite or CI run.

**Subsequent follow-up, October 29:** the [GCC investigation](GCC_WARNINGS.md)
now resolves the four warning groups with fresh local Linux/macOS checks.
That report identifies the changed code and its separate evidence. The audit
snapshot below retains the work and evidence as they stood before that step;
tiny timing reliability was the next scoped step. The subsequent
[timing investigation](TIMING_INVESTIGATION.md) also completes locally, retaining
explicit failure for equal native timestamps.

**Subsequent follow-up, October 30:** the [robustness review](ROBUSTNESS_REVIEW.md)
fills decoding-chunk failure and protected-buffer coverage gaps. Local macOS
Debug/Release/sanitizer, Linux GCC Debug/Release, and native leak checks pass;
no production defect was exposed. [Final M6 verification](M6_VERIFICATION.md)
then passed fresh local checks and all five hosted jobs at `cace0df`, closing
M6 on October 30. Phase 7 release documentation is next. The original audit
table below remains a dated snapshot; these follow-ups record completion of
its scoped work.

The [roadmap checklist](../ROADMAP.md#16-definition-of-done) now distinguishes
completed capabilities from remaining release gates and deferred scope. A
checked capability is evidence of existing behavior, not a claim that the
final release candidate has been published or that all hardening is complete.

## Every previously unchecked requirement

| Original requirement | Audit decision | Implementation, documentation, and existing evidence |
| --- | --- | --- |
| `.cnd` models load from disk | **Complete; check it.** | [`cinder_model_load`](../src/loader.c) decodes and validates the file, constructs layers/model, and publishes ownership only on success. [`model.h`](../include/cinder/model.h) declares it. CTest `model_load`, `model_round_trip_fixtures`, and `mnist_inference_fixtures` exercise real files; [loader](LOADER.md) and [M3](M3_VERIFICATION.md) reports record verification. |
| Tiny exported-network fixture matches each reference layer before MNIST integration | **Complete; check it.** | [`test_model_round_trip.c`](../tests/test_model_round_trip.c) checks saved Dense1, ReLU, logits, and Softmax prefixes for ten inputs: 40 vectors / 120 values. [`check_model_round_trip.py`](../tests/check_model_round_trip.py) also exercises fresh Python export → file → C load → inference. The [round-trip report](ROUND_TRIP.md) and M3 closure precede MNIST integration; M5 reran the comparisons. |
| Input layout and preprocessing are documented and identical in Python and C inputs | **Complete; check it.** | [Format/input contract](MODEL_FORMAT.md#input-file-and-preprocessing-contract), [CLI contract](CLI.md#model-and-input-interpretation), and [`data.py`](../examples/mnist/data.py) specify row-major flattening, float32, `/255`, then little-endian binary32 file values. [`test_mnist_fixtures.py`](../tests/test_mnist_fixtures.py) independently checks every saved input against pixels; [`test_mnist_inference.c`](../tests/test_mnist_inference.c) repeats the pixel/input check in C. [`input.c`](../src/input.c) checks exact lengths and finite values. The runtime consumes preprocessed values; it does not preprocess images itself. |
| Buffer lengths, output capacity, ownership, and error behavior are documented and checked | **Complete within the documented C caller preconditions; check it.** | [Tensor](TENSOR.md), [Dense](DENSE.md), [layer](LAYER.md), [model](MODEL.md), [loader](LOADER.md), and [CLI](CLI.md) contracts match the implementation. `model_run`, `model_allocation_failures`, `loader_failures`, `cli_input`, `cli_audit`, and `cli_setup` cover counts, spare capacity, rollback, caller-output preservation, and cleanup. Valid pointers, true buffer extents, disjoint ranges where required, and caller-managed lifetime/concurrency remain preconditions, not generally detectable errors. |
| `cinder run` works | **Complete; check it.** | [`cli.c`](../src/cli.c) dispatches `run` through [`cli_session.c`](../src/cli_session.c), the input reader, and model API. [DEMO.md](../examples/mnist/DEMO.md) contains actual commands/output. M5 records 133 real process cases, including 11 benchmark cases and the prior 122 CLI cases; MNIST prediction/top-three agreement is 100/100. A read-only sample run was also confirmed during this audit. |
| `cinder info` works | **Not implemented; defer beyond v0.1.0.** | The dispatcher accepts only `--version`, `run`, and `bench`; an actual `info` invocation exits 2 with usage. The opaque model API exposes input/output sizes, not a layer inventory or parameter count. The scope decision below removes this from the v0.1.0 release gate and preserves it in the post-v0.1 backlog. Do not mark it implemented. |
| Loader rejects malformed structures, excessive sizes, and incomplete data without leaking memory | **Complete for the exercised contract; check it.** | [`test_model_load.c`](../tests/test_model_load.c) and [`test_loader_failures.c`](../tests/test_loader_failures.c) cover truncation, types/versions, dimensions/connections, overflow/resource boundaries, non-finite parameters, trailing bytes, and allocation/I/O failures. The [rejection audit](LOADER.md#rejection-and-cleanup-audit--october-24-2026) documents 709 public rejection cases with/without diagnostics and all 21 tiny-model allocation points. M3/M5 lifecycle, sanitizer, and supported leak evidence cover cleanup and recovery. This does not claim arbitrary corrupt pointers are safe or every possible input has been tested. |
| Architecture is documented | **Complete across existing guides; check it.** | [MODEL.md](MODEL.md#working-buffers-and-execution) documents ownership, two reusable buffers, ordered dispatch, failure publication, and concurrency. [LAYER.md](LAYER.md) explains tagged dispatch; [DENSE.md](DENSE.md) and [TENSOR.md](TENSOR.md) describe parameter layout and kernels; [LOADER.md](LOADER.md#validation-and-construction-sequence) explains file-to-constructor flow. The [repository layout](../README.md#repository-layout) maps modules. A concise newcomer-facing architecture summary belongs in the remaining README review; a new architecture file is not a missing runtime feature. |
| README is complete | **Keep open for Phase 7.** | Prerequisites, build/test commands, examples, repository layout, license, and links already exist. The long development history still needs release-oriented editing, supported-command consistency, final version/output updates, and fresh-checkout verification of the final instructions. Correcting selected stale claims in this audit does not finish that review. |
| GitHub `v0.1.0` release exists | **Keep open for Phase 7.** | On October 29 the connected GitHub API returned empty release and tag-reference collections; local `git tag --list` was empty too. CMake still appends `-dev`, and the current binary prints `Cinder 0.1.0-dev`. Version finalization, release notes, final candidate checks, tag, and publication remain work to do. No repository publication was performed by this audit. |

## Scope decision: defer `cinder info` beyond v0.1.0

**Decision: v0.1.0 ships `--version`, `run`, and `bench`. Model inspection via
`cinder info` is post-v0.1 work, with no committed delivery date.** This is an
explicit revision of the original plan, not a completed feature or an accidental
omission. The roadmap's opening example and release feature list now reflect
this decision; the proposed inspection example is preserved in the backlog.

The release's central workflow already exists: export supported sequential
float32 models, load their parameters in C, run Dense/ReLU/Softmax inference,
compare with Python, and benchmark execution. `info` would improve convenience
but is not required for any of these steps. Keeping it out also preserves the
feature-freeze intent while Phase 6 reviews correctness.

The proposed output includes layer types/shapes and a parameter total. Loaded
models are opaque; the public model accessors currently provide only external
input/output counts. Implementing inspection properly therefore requires a
metadata-access design (public or private), an output/error contract, checked
parameter-count arithmetic, malformed-model handling, and CLI tests including
output failures. Copying the binary parser into the CLI or exposing mutable
model internals just to print a report is not an appropriate shortcut.

When revisited, the scoped work should be: define inspection semantics and
metadata access; reuse the validated loader; implement read-only reporting;
then test dimensions/counts, unusual filenames, unsupported/malformed files,
cleanup, and output failures. The existing `.cnd` v1 byte format need not change
merely to report metadata already derivable from valid layers.

## Actual work remaining before release

This table preserves the October 29 audit plan. Its four Phase 6 rows are now
complete; [M6 verification](M6_VERIFICATION.md) records their results. The three
Phase 7 rows remain open, with final candidate checks after later changes.

| Work | Completion evidence needed | Phase |
| --- | --- | --- |
| Triage GCC warnings | Reproduce `-Wtype-limits` loader guards and Release `-Wmaybe-uninitialized` diagnostics in source-including tests. Record the cause and disposition of each; fix demonstrated defects and verify any changes. A passing suite alone does not resolve a warning. | 6, next scoped step |
| Investigate tiny-model timing reliability | Investigate the two saved zero-interval failures. Record whether there is a correctable defect or a documented clock/workload limitation. Retain explicit failure behavior unless a separately defined measurement change is justified; a batching redesign is not automatically required for release. | 6 |
| Review remaining robustness coverage | Map the existing failure tests to the robustness list, identify concrete gaps, and add focused tests/fixes where warranted. Do not assume previous passing tests prove exhaustive coverage. | 6 |
| Verify and close M6 | Run Debug/Release, numerical/lifecycle checks, sanitizers, supported leak checks, and all macOS/Linux CI jobs on the resulting code revision; document findings and accepted limitations. Prior M5 passes remain evidence for that revision, not automatic approval of future changes. | 6 |
| Finish the release README and instructions | Present the supported commands, architecture/ownership overview, reproduction commands, measured results, and limitations clearly. Verify the documented workflow from a fresh checkout of the release candidate. | 7 |
| Finalize version and release notes | Make the release executable report `0.1.0` consistently with its tests/docs; CMake currently unconditionally appends `-dev`. Prepare accurate scope/results/limits and verify the final candidate after these edits. | 7 |
| Tag and publish v0.1.0 | Publish only after the gates pass and the user authorizes the necessary commits/push/tag/release actions. Confirm the published tag and release identify the verified revision. | 7 |

Existing documented limits remain: caller serialization for a shared model,
no authenticity/checksum in `.cnd` v1, saved-fixture rather than full-test C
accuracy, tiny per-call timing failures, and no completed hosted native macOS
leak check. Local native checks and hosted Linux LeakSanitizer passed. These
are recorded constraints, not newly discovered implementations to add here.

## Evidence checked during this audit

Source review covered the CLI dispatcher/session/input reader, loader cleanup
and constructor path, opaque model representation, shape/capacity validation,
working-buffer loop, and public contracts. Relevant tests were inspected and
confirmed registered in CMake; `ctest --show-only=json-v1` lists 37 ordinary
C tests. The original ten unchecked requirements were enumerated before edits.

The retained Release executable's SHA-256 was checked against the M5 baseline:
`766f38969630db0335468d89658559660337ec24167ae23e8c2ecea03aed4e05`.
Three commands then ran without rebuilding or changing source artifacts:

| Command, after `./build-baseline-release-20261028/cinder` | Exit | Observation |
| --- | ---: | --- |
| `--version` | 0 | `Cinder 0.1.0-dev`, empty stderr |
| `run examples/mnist/mnist.cnd tests/fixtures/mnist/inputs/00000.f32` | 0 | Prediction 7; ranked indices 7, 3, 2; empty stderr |
| `info examples/mnist/mnist.cnd` | 2 | Empty stdout; `cinder: usage: expected --version, run, or bench.` followed by the existing usage block |

Read-only GitHub checks on October 29 used
[the releases collection](https://api.github.com/repos/farukumarx64/cinder/releases?per_page=100)
and [tag references](https://api.github.com/repos/farukumarx64/cinder/git/matching-refs/tags/).
Both returned `[]`; no published release or tag was reported. This is dated
state, not a promise about later publication.

This step changes only documentation and planning. It does not implement
`info`, fix compiler/timer behavior, rerun the full test matrix, close M6/M7,
change the version, create tags, or publish a release. Documentation checks
cover links, checklist reconciliation, consistent command scope, and whitespace.
