# C loader — contract and implementation

**Contract and initial implementation completed October 24, 2026.**
The declaration is in [`include/cinder/model.h`](../include/cinder/model.h), and
the shared status codes are in [`include/cinder/error.h`](../include/cinder/error.h).
[`src/loader.c`](../src/loader.c) implements the linkable C API. The CLI still
supports only `--version`; loading through CLI commands is later work.
The [format specification](MODEL_FORMAT.md) remains authoritative for file bytes
and resource bounds; this document defines the C API, file handling, ownership,
failure behavior, and use of the existing constructors.

## Public function and caller requirements

```c
#include "cinder/model.h"

cinder_status_t cinder_model_load(
    const char *path,
    cinder_model_t **out_model,
    cinder_error_t *error
);
```

The function loads one nonempty sequential `.cnd` version 1 model. It accepts a
path to a seekable binary file; it does not take input vectors or run inference.
The returned object uses the existing size getters, `cinder_model_run`, and
`cinder_model_free`. No separate loaded-model type or destructor is introduced.

- `out_model` is a writable pointer slot that must not already own a live model.
  Initialize the caller's variable to `NULL`. The loader does not inspect its
  previous value or free/replace a previous model.
- `path` is a borrowed, readable NUL-terminated string, unchanged during the
  call. `NULL` and an empty string are invalid. Its bytes follow the host's file
  naming conventions. Relative paths use the process working directory; the
  caller must not race a working-directory change against relative loading.
- `error` is optional caller-owned storage. The output slot, path bytes including
  their terminator, and error storage must be disjoint and live for the call.
  Pointer validity and overlap are caller preconditions, not detectable errors.
- The `.cnd` suffix is conventional; the bytes determine validity. `"-"` is an
  ordinary filename, not standard input. There is no `FILE *`, descriptor,
  memory-buffer, archive, URL, or non-seekable streaming interface in this step.

If `out_model == NULL`, return `CINDER_ERROR_INVALID_ARGUMENT`, setting optional
error details. Otherwise clear `*out_model` before checking any other argument,
opening a file, or allocating model data. Every later failure leaves it `NULL`.

## Returned ownership and lifetime

On `CINDER_OK`, the caller exclusively owns a fully constructed model. The model
owns its ordered layer list, every layer, and two reusable working buffers;
each Dense owns its copied weight/bias tensors. Call `cinder_model_free` exactly
once when finished. `cinder_model_free(NULL)` remains safe after a failed load.

The loader retains no path pointer, file handle, raw file bytes, or temporary
parameter array. After success, the caller may release the path or change,
replace, or delete the file without changing the model. Loading the same path
twice creates independent layers, parameters, and work buffers; there is no
cache or shared parameter owner.

A failed load cannot invalidate a separately owned model. Retrying requires
no reset. To replace an existing model, load into a separate initially NULL
candidate, then free/swap the old owner only after success.

Concurrent loads use independent local state and must have distinct writable
output/error storage. They may read the same stable file. The resulting models
retain all [existing execution rules](MODEL.md): one active inference per model,
caller serialization for shared instances, and no destruction during use.

## Status codes and diagnostics

Four status values are appended after the existing codes, preserving values
`0..7`: `CINDER_ERROR_IO`, `CINDER_ERROR_FORMAT`,
`CINDER_ERROR_UNSUPPORTED`, and `CINDER_ERROR_LIMIT`. They are runtime statuses,
not serialized format values. Existing `cinder_error_t` storage remains unchanged.

| Status | Loader meaning |
| --- | --- |
| `CINDER_OK` | Complete model published; file closed and all temporary storage released. |
| `CINDER_ERROR_INVALID_ARGUMENT` | NULL output slot, NULL path, or empty path. |
| `CINDER_ERROR_IO` | Opening, seeking, determining size, reading, or closing fails. A non-seekable source also fails here. Failures inside stdio use this status even if the host reports a resource shortage. |
| `CINDER_ERROR_FORMAT` | Bad magic; zero layer count/dimension; zero or unknown layer ID; wrong payload length; insufficient declared-record space; truncated header/payload; any trailing byte. |
| `CINDER_ERROR_UNSUPPORTED` | Magic identifies Cinder but version is not 1 (including 0), or the host cannot decode the required binary32 object representation. Never guess a version or skip an unknown record. |
| `CINDER_ERROR_LIMIT` | A positive layer/vector count exceeds its v1 cap, cumulative parameter bytes exceed 64 MiB, or file size exceeds 67,125,264 bytes. |
| `CINDER_ERROR_OVERFLOW` | A required native count, allocation extent, offset, or conversion is not representable after applicable format bounds are checked. Distinct from a representable value exceeding the format policy. |
| `CINDER_ERROR_DIMENSION_MISMATCH` | Header input differs from the first layer input, adjacent layers disagree, or an activation changes vector length. |
| `CINDER_ERROR_NONFINITE` | Any encoded weight/bias is NaN or either infinity. Signed zero, subnormals, and finite extremes are valid. |
| `CINDER_ERROR_ALLOCATION` | An explicit loader allocation or existing layer/model constructor allocation fails. |

Constructor failures retain their returned status. File metadata is checked
before constructor calls, so argument errors from those calls indicate a broken
implementation invariant rather than a substitute for parsing checks. Do not
translate every constructor failure into `FORMAT`. `OUT_OF_BOUNDS` and
`BUFFER_TOO_SMALL` are not ordinary loader validation outcomes.

When supplied, `error->code` matches the returned status, and `error->message`
is empty on success or bounded and NUL-terminated on failure (capacity 128).
Messages identify the stage/field, and a zero-based layer index and absolute
byte offset when available, such as `Layer 2 at byte 96: input size mismatch.`
Constructor diagnostics may receive a bounded context prefix without changing
the status or discarding their cause. Wording is not a machine-readable API.
No diagnostic allocation, printing, process termination, or global last-error
state is introduced. The caller must not use `errno` as Cinder's error result.

Keep the first failure and its diagnostic during cleanup. If parsing fails and
`fclose` also fails, return the parsing failure. A close failure after otherwise
successful parsing returns `IO`; no model is published. When a read reports
both a short count and an error indicator, `IO` takes priority over truncation.
With multiple independent file violations, callers must not depend on one
particular diagnostic; the checks below establish the safe construction order.

## File handling and host representation

Open the path once with `fopen(path, "rb")`; use that stream for both size
discovery and reading. Normal host path resolution, including symlinks, applies.
No write, truncation, rename, or file-content modification is performed. The
loader owns the stream; the caller never receives or closes it.

This initial API targets seekable binary files on the supported macOS/Linux
toolchains. Seek to the end, obtain a nonnegative byte length, and return to
the beginning, checking every operation. Failed seek/size discovery is `IO`;
there is no fallback that reads an unbounded stream. A successfully measured
file beyond the format cap is `LIMIT`. Check sizes before narrowing to native
allocation types. The whole-file cap fits even a 32-bit signed file position.

The size check establishes available byte ranges before model-data allocations;
it never replaces checking actual reads. Use byte-oriented exact reads into
byte buffers, not direct reads into runtime structs or float arrays. For an
incomplete read, `ferror` means `IO`, clean early EOF means `FORMAT`, and a
short read with neither indication is `IO`. At the final boundary, require the
parsed length to equal the measured length and probe for clean EOF: an extra
byte is `FORMAT`, a read error is `IO`, and only clean EOF is acceptable.

The caller must keep the opened file's contents stable throughout the call.
The loader does not lock it, reopen it, map it, or guarantee a snapshot against
concurrent in-place writers. Reads are blocking; the API has no timeout or
cancellation mechanism. A seekable file may still encounter later I/O errors.

The tensor implementation already has compile-time requirements for eight-bit
bytes and binary32 precision/range. Loading additionally requires compatible
IEEE binary32 object representation, including signed zeros and subnormals.
The implementation must establish that representation before opening the file;
an unsupported representation on an otherwise buildable host returns
`UNSUPPORTED`. This does not remove the existing compile-time requirements.

Decode each `u32le` from bytes using unsigned arithmetic. For parameters, decode
the 32-bit word, reject an all-ones exponent (NaN/infinity), and transfer valid
bits into aligned `float` storage without aliasing violations (for example,
`memcpy` on a verified compatible host). Do not reinterpret unaligned file
bytes as native pointers or use arithmetic conversion that loses signed zero
or subnormal bits. Host endianness must not change model values or the file
layout; unsupported mixed integer/float layouts must be rejected explicitly.

## Validation and construction sequence

1. Validate output/path arguments in the order above and establish the host
   representation. No file or model allocation is needed for these checks.
2. Open once, determine the size, and reject a file above the whole-file cap
   with `LIMIT`, or one shorter than the 16-byte fixed header with `FORMAT`.
3. Read the complete header. Check magic, exact version, positive bounded input
   size and layer count, then ensure enough bytes remain even for the declared
   record headers. Check native representability before dependent operations.
   The limits remain 1,024 layers and 65,536 floats per vector.
4. Allocate a temporary ordered layer-pointer array with checked native size;
   initialize every slot to a semantic NULL pointer. Only loader-created,
   distinct layer objects are placed there.
5. Visit records in order. Check availability/read of the full 16-byte header,
   ID, positive bounded dimensions, connection to the header/previous record,
   and activation size preservation. Compute exact parameter counts and bytes
   using checked wide arithmetic. Require the payload field to match the derived
   length, enforce the cumulative 64 MiB parameter cap, check native extents,
   and establish the full payload lies within the measured file before staging
   allocation or reads. File lengths never authorize unchecked allocation.
6. For Dense, allocate one aligned temporary float array for that record's
   weights followed by biases. Decode/check the complete payload, then call
   `cinder_layer_create_dense(input_size, output_size, weights, weight_count,
   bias, bias_count, &layers[index], error)`. The constructor copies parameters;
   release staging after the call, whether it succeeds or fails. Only one
   record's staging exists at a time. On decode failure, release it without
   calling the constructor. For activations, map file ID 2/3 explicitly to
   `CINDER_LAYER_RELU`/`CINDER_LAYER_SOFTMAX` and use
   `cinder_layer_create_activation(type, size, &layers[index], error)`.
7. After the declared records, require exact EOF as described above. Close the
   stream and check the result. Treat the stream as consumed after this single
   `fclose` attempt, including failure; never retry or use it again.
8. With the file closed, call `cinder_model_create(layers, layer_count,
   &candidate, error)` using a private initially NULL candidate. The existing
   constructor revalidates metadata/connections, determines the largest vector,
   and allocates its pointer list and two reusable working buffers.
9. On model-construction success, it owns every layer and has cleared the
   temporary slots. Free the now-empty temporary pointer array, then publish
   `*out_model = candidate` and clear error details. No fallible work remains.

Do not allocate or populate `cinder_model_t`, `cinder_dense_t`, or tensor owners
manually. Using the constructors preserves their tested validation, copying,
rollback, and model-transfer rules. Loading never calls layer forward methods.
For the tiny fixture, both final model work buffers have capacity four floats,
exactly as in ordinary in-memory construction.

The 64 MiB cap covers file parameters, not peak memory. Already constructed
layers, the current staging array, constructor copies, pointer lists, stream
buffering, and model workspaces may coexist. Avoid a whole-file allocation;
check every individual allocation and release every partial owner.

## Cleanup responsibilities

| State reached | Owner and cleanup on failure |
| --- | --- |
| File opened | Loader makes one `fclose` attempt. Preserve an earlier failure if closing also fails. |
| Temporary layer list allocated | Loader frees all non-NULL slots with `cinder_layer_free`, then frees the pointer array. |
| Dense staging allocated | Loader frees it on decode failure and after the Dense constructor returns. |
| Layer constructor fails | Constructor releases its own partial objects and leaves its output NULL. Loader still frees staging and previously completed layers. |
| Model constructor fails | Constructor releases its new partial model allocations; loader still owns every unchanged layer slot and frees them. |
| Model constructor succeeds | Model owns the layers; cleared temporary slots prevent double frees. Loader frees only the temporary pointer array and publishes the candidate. |
| Load succeeds | Caller owns the model and later calls `cinder_model_free`. No loader file/staging/list storage remains. |

Every normal returned failure releases all library-owned partial allocations
and returns no model. Standard stream handling requires discarding the stream
after a close attempt even if closing reports failure. The contract guarantees
one close attempt and no retained stream; it cannot make a failed operating
system close succeed. It does not cover abrupt process termination or invalid
caller pointers. The `fclose` and short-read rules follow C11 draft N1570
[sections 7.21.5.1 and 7.21.8.1](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf).

## Caller example

```c
#include "cinder/model.h"

#include <stdio.h>

int main(void)
{
    cinder_model_t *model = NULL;
    cinder_error_t error;
    cinder_status_t status = cinder_model_load(
        "tests/fixtures/tiny/tiny.cnd", &model, &error);
    if (status != CINDER_OK) {
        fprintf(stderr, "Load failed (%d): %s\n", (int)status, error.message);
        return 1; /* model is NULL; the loader already cleaned up. */
    }

    if (cinder_model_input_size(model) != 2 || cinder_model_output_size(model) != 2) {
        cinder_model_free(model);
        return 1;
    }
    const float input[] = {2.0f, -1.0f};
    float output[2];
    status = cinder_model_run(model, input, 2, output, 2, &error);
    if (status == CINDER_OK) {
        printf("%.9g %.9g\n", (double)output[0], (double)output[1]);
    } else {
        fprintf(stderr, "Run failed (%d): %s\n", (int)status, error.message);
    }
    cinder_model_free(model);
    return status == CINDER_OK ? 0 : 1;
}
```

Printing here is caller behavior; the library itself does not print. The shape
check protects the fixed two-element example buffers. This produces the saved
probabilities within the existing comparison tolerance.

## Contract verification — October 24, 2026

- Both changed headers passed standalone and repeated-inclusion syntax checks
  with `-std=c11 -Wall -Wextra -Wpedantic -Werror`.
- Both C blocks above, including the complete caller example, passed the same
  syntax checks. At the contract step, the example was not linked or run because
  the loader was only declared.
- Compile-time checks confirmed the function's exact pointer type, unchanged
  existing status values `0..7`, and appended status values `8..11`.
- The ordinary Debug C build succeeded, and **19/19 existing CTest checks
  passed** after rebuilding against the changed headers. These are regression
  checks for implemented runtime behavior, not loader tests.

No parser implementation, new runtime tests, CMake target, or sanitizer/hosted
CI result was added by this contract step. File bytes and existing runtime
implementations are unchanged.

## Implementation details

The parser keeps a stream, measured size, byte offset, and current layer index
in local state. Explicit unsigned shifts decode little-endian integers. Host
representation probes compare known float encodings (including negative zero,
subnormals, and finite extremes) before opening a file. Parameter words with an
all-ones exponent fail before conversion; `memcpy` transfers other words into
aligned float storage without unaligned accesses or pointer aliasing.

The format bounds make dimension products and byte counts safe in `uint64_t`.
Only after validating those counts does the loader narrow to `size_t`, checking
native allocation extents first. Remaining-file checks use subtraction to avoid
overflow in an end-offset addition. No whole-file buffer is allocated. Dense
payloads are decoded with a 4 KiB byte scratch buffer into one temporary float
array, then copied by the existing Dense constructor and immediately released.

Layer constructors receive only validated metadata and decoded parameters.
Activations map file IDs explicitly to runtime tags. After all records, the
loader checks the measured boundary, probes clean EOF, and closes the stream.
It then calls `cinder_model_create` to adopt the layers and allocate the existing
two working buffers. The caller receives the model only after this succeeds.

One cleanup path frees parameter staging, constructed layers, and the temporary
pointer list, and attempts to close any still-open stream. Constructor failures
retain their status and diagnostic cause. An earlier failure survives cleanup;
a close failure on the successful parsing path prevents model publication.

## Verification status

The rejection/cleanup audit below now covers parser categories, exact resource
boundaries, parameter-bit preservation, and forced allocation/stdio failures.
The [full export/load/inference comparison](ROUND_TRIP.md) now passes against
every saved stage, both from committed files and fresh Python exports. The
[loaded-model lifecycle audit](M3_VERIFICATION.md) also passes locally, including
Debug/Release, sanitizers, and native leak checks. All three hosted macOS/Linux
jobs passed for `5ad7dc4`, closing M3 on October 25, 2026.

Local results below describe only executed checks. Big-endian execution and
unsupported host representations have not been exercised. Same-file mutation
during loading remains outside the caller contract.

## Implementation verification — October 24, 2026

[`tests/test_model_load.c`](../tests/test_model_load.c) is registered as
`model_load` in CTest and links the normal `cinder_core` library. It reads the
committed binary fixture and writes disposable files only in its build-directory
test folder; no Python installation is needed for this C test.

Successful-load checks cover:

- The saved 168-byte `2 -> 4 -> 2` network, with ten deterministic inputs and
  the shared shape, finite-value, and `atol = rtol = 1e-5` comparison checks.
  Five passes across independent loads, file deletion, and failed-load recovery
  compare **50 output vectors / 100 values**. Maximum absolute error:
  **`5.96046448e-8`**. These compare final probabilities; the integrated file
  round trip of every intermediate stage was added in the [next step](ROUND_TRIP.md).
- ReLU-only and Softmax-only files, an independently encoded non-square `2 -> 3`
  Dense, and a `1 -> 1025` Dense whose payload spans three read blocks.
- Acceptance of positive/negative zero, subnormal encodings, and finite extremes.
  Acceptance was tested in the implementation step; the later audit below adds
  exact internal parameter-bit inspection.
- Input preservation, untouched spare output capacity, independent model owners,
  continued use after deleting the source file, no modification of input files,
  and successful loading after earlier failures.

At the initial implementation step, the test exercised **206 rejection cases**, each with and without optional error
storage: every one of the tiny fixture's 168 truncated prefixes, malformed
magic/version/type/dimension/connection/payload fields, parameter NaNs/infinities
(including a later decoding block), trailing zero/nonzero bytes, representative
resource-limit and arithmetic-wrap attempts, invalid arguments, and a missing
file. Rejections check status, bounded diagnostics, and NULL model output.
Later-record failures exercise cleanup after earlier layers were constructed.

Actual local checks on Apple Silicon with Apple Clang 21.0.0 / CMake 4.4.4:

| Configuration/check | Result |
| --- | --- |
| Debug, fresh NumPy references and both Python exporter suites enabled | 24/24 CTest checks passed |
| Release, ordinary C-only suite | 20/20 passed |
| Combined ASan/UBSan Debug, ordinary C-only suite | 20/20 passed; no sanitizer diagnostics |
| New source and test, C11 with `-Wall -Wextra -Wpedantic -Werror` | Passed |
| Caller example above, compiled/linked/run | Printed `0.904650509 0.0953494608` |
| macOS `leaks --atExit` on the loader test | Exit 0; 0 leaks / 0 leaked bytes reported |

The native leak tool required execution outside the sandbox to inspect its child
process. It also reported the normal restricted-process memory-content warning;
the zero-leak result is evidence for this executed test, not a complete allocation
failure audit. LeakSanitizer was not enabled on macOS. The existing Linux CI job
enables it. Hosted CI was pending at this implementation step and subsequently
passed at [M3 closure](M3_VERIFICATION.md). The C test runs in the existing
macOS, Linux, and Linux sanitizer jobs.

To run the new test after configuring and building normally:

```sh
ctest --test-dir build -R '^model_load$' --output-on-failure -V
```

## Rejection and cleanup audit — October 24, 2026

This step expands `test_model_load.c` and adds
[`tests/test_loader_failures.c`](../tests/test_loader_failures.c). The production
loader and public API needed no changes. The ordinary test links `cinder_core`;
the audit compiles the same real loader/tensor/Dense/layer/model sources with
allocator and stdio substitutions confined to its own translation unit. It
reuses the complete ordinary test suite so malformed cases run against both the
normal library and tracked implementation. There is no production fault switch.

### Parser and resource coverage

The ordinary suite now has **709 rejection cases**, each run with and without
error storage, plus its successful loads and 50 repeated NumPy comparisons.
The additions cover:

- Unknown/zero type IDs at every layer position; additional unsupported version
  words; zero, over-limit, and `UINT32_MAX` input/output dimensions in every record;
  incorrect payload byte counts, including lengths not divisible by four.
- All nine Dense/ReLU/Softmax adjacency combinations with mismatched connections.
- Eight non-finite encodings at each of the 22 tiny-network parameter positions:
  both infinities, both signs of quiet/signaling NaNs, and varied NaN payloads.
- All 256 possible one-byte suffixes and concatenated valid models. Existing
  coverage retains every truncated prefix of the tiny fixture and arithmetic
  wrap attempts. Unknown record types return `FORMAT`; versions other than one
  return `UNSUPPORTED`, following the contract.
- A valid 1,024-layer activation model with 65,536-element vectors, and a fully
  present 1,025th layer rejected by the count limit.
- Real Dense payloads with cumulative parameter sizes of **67,108,860**,
  **67,108,864**, and **67,108,868** bytes: one float below the budget, exactly at
  it, and one float above. The over-budget case has individually valid layers
  and enough file bytes; it fails only after the first large Dense exists.
- A file at the **67,125,264-byte** maximum with 64 MiB of parameters and 1,024
  layer headers. Removing its last header byte yields `FORMAT` at max minus one;
  adding one byte to the valid maximum yields `LIMIT` at max plus one.

Large fixtures are written in bounded zero-filled chunks, loaded through the
real file API, and removed after the tests. They are generated in separate
ignored build directories, never committed. The largest Dense requires about
128 MiB for staging plus owned parameter copies while loading; sanitizer and
allocator bookkeeping can increase process memory. CTest gives each executable
60 seconds to accommodate these boundary cases on CI.

### Allocation ownership and forced failures

The audit tracks every explicit `malloc`, `calloc`, and `free` in the complete
runtime ownership chain. Each failing call must leave exactly the same existing
live blocks and byte total as before the call, a NULL output model, and no open
loader stream. Any untracked/double free fails immediately. Successful model
destruction must also return the allocation ledger to zero.

The saved tiny network makes **21 explicit allocation attempts**:

| Owner | Attempts |
| --- | --- |
| Loader's temporary layer-pointer list | 1 |
| Two Dense staging arrays | 2 |
| Two Dense layers, including tensor structs/data and wrappers | 12 |
| ReLU and Softmax wrappers | 2 |
| Final model, private pointer list, and two working buffers | 4 |

Each attempt is forced to fail individually, both with and without diagnostics.
Checks require `ALLOCATION`, an immediate stop at the selected attempt, balanced
successful allocations/frees, one file close, no published model, and successful
reload/inference afterward. The four model-construction failures occur after
the file has closed and exercise cleanup of the complete layer sequence.

### File errors, diagnostic preservation, and exact bits

There are **29 injected file-failure scenarios**, each with and without error
storage and followed by successful reload/inference:

- Open, both seeks, size discovery, and final close.
- Each of the tiny model's seven reads returning short with an error indicator,
  clean EOF, or neither indicator. The error case sets both error and EOF flags
  to confirm that `IO` takes precedence over truncation.
- Final EOF probing with an error, an extra byte, or unconfirmed EOF.

The close wrapper actually consumes the stream before reporting an injected
failure. This detects retries, double closes, and use of a closed stream. The
model-constructor observer requires the stream to be closed before adoption.
Pairing a late malformed record or each of the 17 pre-close allocation failures
with a close failure confirms that the original status and message survive.

Native allocation-extent overflow is unreachable for valid v1 files on the
supported 32/64-bit size ranges because the format caps are smaller. Three
test-only lower `SIZE_MAX` ceilings exercise the loader's vector, pointer-list,
and parameter-staging overflow guards, with and without error storage. This
tests guard behavior and cleanup; it does **not** claim execution on a different
ABI. Actual malformed large dimension products still run with native bounds.

Private test-only inspection verifies exact weight/bias bits for 12 finite
encodings, including both zeros, signed subnormals, normal boundaries, asymmetric
bit patterns, and signed finite extremes, after staging and the source file are
gone. No public model-inspection API was added. Successful recovery inference
also confirms zero allocation/free attempts during execution.

In total, the audit checks cleanup after **1,562 rejected calls**, including the
shared parser cases, allocation failures, file failures, combined failures, and
simulated native-size guards. The ledger covers explicit runtime allocations;
stdio's internal allocations remain covered by normal close behavior and the
external sanitizer/leak checks, rather than a claim to intercept libc internals.

### Executed verification

On October 24, 2026, Apple Silicon / Apple Clang 21.0.0 / CMake 4.4.4:

| Check | Result |
| --- | --- |
| Debug with fresh NumPy references and both exporter suites | **25/25 CTest checks passed** |
| Release, C-only suite | **21/21 passed** |
| Combined ASan/UBSan Debug, C-only suite | **21/21 passed**, no sanitizer diagnostics |
| Both changed test translation units with C11 warnings as errors | Passed |
| Native macOS `leaks --atExit`, ordinary loader test | Exit 0; **0 leaks / 0 leaked bytes** |
| Native macOS `leaks --atExit`, tracked failure audit | Exit 0; **0 leaks / 0 leaked bytes** |

Both native leak runs required execution outside the sandbox for child-process
inspection and reported the restricted-process memory-content warning described
above. LeakSanitizer was not enabled on macOS. These tests are registered for
the existing macOS/Linux CI jobs and Linux ASan/UBSan/leak job. Hosted verification
was pending at this audit step and subsequently passed at [M3 closure](M3_VERIFICATION.md).

Run both loader tests after building with:

```sh
ctest --test-dir build -R '^(model_load|loader_failures)$' --output-on-failure -V
```

The [complete round trip](ROUND_TRIP.md) now compares every saved intermediate
stage. The [M3 report](M3_VERIFICATION.md) records the subsequent loaded-model
lifecycle checks, expanded allocation audit (1,604 rejected calls), and local
results, including successful hosted CI and M3 closure. Big-endian execution,
a different native ABI, and unsupported-host representation behavior remain
unexecuted platform paths; file mutation during loading remains outside the
caller contract.
