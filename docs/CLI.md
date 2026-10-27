# `cinder run` command contract

**Defined October 26, 2026; implemented October 27, 2026.**
[`src/cli.c`](../src/cli.c) supports `--version` and `run`, using the private
[`input reader`](../src/input.c) and public model API. See the
[demonstration and verification report](../examples/mnist/DEMO.md) for actual
output, reproduction commands, test results, and M4 closure status.

The [.cnd format](MODEL_FORMAT.md), [loader](LOADER.md), and [model API](MODEL.md)
remain authoritative for binary encoding, validation, ownership, and inference.

The separate [benchmark contract](BENCHMARK.md) defines the planned `bench`
command. It is not implemented yet. Its future integration will add a third
line to the common usage block and change the global unknown-command reason;
the two-command usage text below describes the current executable. Successful
`run`/`--version` output and `run` operational errors will retain this contract.

## Invocation and argument rules

```text
cinder --version
cinder run <model.cnd> <input.bin>
```

- `--version` remains an exact one-argument command. It prints
  `Cinder <build-version>\n` to stdout, leaves stderr empty, and exits `0`.
- `run` requires exactly two nonempty positional path arguments, in model/input
  order. Command names are case-sensitive. Wrong counts, empty paths, missing
  commands, and unknown top-level commands are usage errors.
- Paths use the process working directory unless absolute. The shell handles
  quoting; quote a path containing spaces so it remains one argument. Ordinary
  host path resolution, including symlinks, applies.
- `.cnd` and `.bin` are descriptive suffixes, not filename validation rules.
  The saved `.f32` fixtures are valid input paths. Bytes determine validity.
- There is no option parsing after `run`: both following arguments are literal
  paths, even when they begin with `-`. A path named `-` is an ordinary file,
  not stdin. `--help`, `--`, `--top-k`, batching, and output-file options are not
  part of this initial command.
- One invocation loads one model and executes exactly one input vector. It
  does not modify either source file or create result files.

Run the saved fixture from the repository root:

```sh
./build/cinder run examples/mnist/mnist.cnd tests/fixtures/mnist/inputs/00000.f32
```

## Model and input interpretation

Accept every model supported by `cinder_model_load`, including models without
a final Softmax and output sizes other than ten. Do not infer semantics from
the filename, append an activation, normalize scores, or inspect opaque model
internals. The existing input/output size accessors are sufficient.

For model input size `D`, the input file contains exactly `4 * D` bytes:

- `D` little-endian IEEE binary32 floats in feature-index order.
- No header, padding, extra samples, or trailing bytes.
- All values finite; signed zeros and subnormals are permitted.
- Already preprocessed by the caller. No image decoding, flattening, scaling,
  or `[0,1]` restriction is applied by the CLI.

The MNIST example supplies 784 row-major pixels, cast to float32 and divided by
float32 `255.0`: exactly 3,136 bytes. Its preprocessing is an example convention,
not a property detected from `.cnd` bytes.

Use seekable binary files, with stable contents during each read. The model
loader's existing size and resource bounds apply. For inputs, check the expected
byte extent before allocation; measure length on the opened stream and require
exact agreement before decoding. Check every read, reject the first non-finite
value in index order, and confirm clean EOF and successful close before inference.
Non-seekable input is an I/O error; no unbounded-stream fallback is provided.

## Class-index mapping and ranking

For output length `M`, output position `i` is class index `i`, in `0..M-1`.
These are zero-based unsigned decimal indices. `.cnd` v1 contains no class names
or label map; the application/exporter documentation defines their meaning.
For this MNIST checkpoint, index `0` means digit 0, index `1` means digit 1,
and so on through index `9`. Renaming the model does not change that mapping.

The predicted index is the position of the greatest **raw final model value**.
Display the best `K = min(3, M)` entries in descending score order. Exact numeric
ties use the smaller index first, including a tie between positive and negative
zero. Compare the full float values before formatting. Never repeat an index,
invent classes to fill three rows, or reorder the output-to-label mapping.
A one-output model therefore predicts index 0 and prints one score row.

Scores are printed as raw values, without a percent sign. With inference Softmax,
as in MNIST, these values are probabilities. Without Softmax, they may be logits
or other finite scores, including negative values. The command does not claim
that every valid model produces probabilities or that probabilities are calibrated.

## Exact successful stdout format

```text
Model: <escaped-basename>
Prediction: <winning-index>
Scores:
<index><TAB><score>
...exactly K score rows...
```

The ellipsis above describes repetition; it is never printed. Formatting rules:

- Model display uses the basename of the supplied model path, including its
  extension. On supported macOS/Linux hosts, use the component after the last
  `/`. Do not resolve a symlink to choose the displayed name.
- Header capitalization and spaces are exactly as shown. Each score row uses
  one literal tab (`0x09`) between its two fields. Every line ends with LF,
  including the last; there are no blank lines or trailing spaces.
- Print indices as ordinary unsigned decimal integers without leading zeros.
- Print scores using C `%.9g` after promotion to `double`, under the `C` numeric
  locale. This means nine significant digits, trailing zeros removed, and
  scientific notation where `%g` selects it. Use a decimal point, no grouping
  separators, and normalize either signed zero to the displayed text `0`.
- Print no timing, progress, accuracy, banners, or additional model metadata.
  Timing belongs to the later benchmark command. Success leaves stderr empty.

Actual local C output for MNIST test index 0, captured October 27, 2026.
Cross-platform numerical checks use the fixture tolerance; the final printed
digits need not be identical on every host. Each score row contains one tab:

```text
Model: mnist.cnd
Prediction: 7
Scores:
7	0.9978019
3	0.0018843736
2	0.000236767111
```

Actual output for the tiny worked fixture has only two score rows:

```text
Model: tiny.cnd
Prediction: 0
Scores:
0	0.904650509
1	0.0953494608
```

If a model returns `[0.5, 0.5, 0.1, 0.5]`, prediction is 0 and the displayed
indices are `0, 1, 3`. If it returns `[-0.5, -2.75]`, prediction is 0 and the
printed scores stay negative; the CLI does not apply Softmax.

## Exit statuses and stream behavior

| Exit status | Meaning |
| ---: | --- |
| `0` | Version output completed, or model loading, input validation, inference, and prediction-output flush all succeeded. |
| `1` | Operational failure: file I/O, invalid/unsupported model or input, resource/size limit, allocation, inference, or output failure. |
| `2` | Invalid invocation: command, argument count, or empty path. |

These are CLI process statuses, independent of `cinder_status_t` enum values.
Do not return a library enum directly as the process status. The table covers
normal program exits; host signal termination is outside this mapping.

Usage errors write the diagnostic and usage block below to stderr and nothing
to stdout. Other failures write one diagnostic line to stderr without usage.
No prediction text, including the `Model:` header, is written before model/input
validation and inference finish successfully. Consequently, failures before
printing leave stdout empty.

Check output writes and the final stdout flush. A detected output failure exits
`1` and attempts the output diagnostic; stdout may already contain a partial
result and cannot be rolled back. Failure to write stderr must not turn failure
into success. This does not promise atomic writes to terminals, pipes, or files.

## Usage diagnostics

The complete stderr block is:

```text
cinder: usage: <reason>
Usage:
  cinder --version
  cinder run <model.cnd> <input.bin>
```

Each line ends with LF, with no extra blank line. Select the reason in this order:

| Condition | Exact reason |
| --- | --- |
| No command, an unknown top-level command, or `--version` with extra arguments | `expected --version or run.` |
| Command is `run`, but not exactly two path arguments | `run requires exactly two file paths.` |
| Correct argument count but model path is empty | `model path must not be empty.` |
| Model path nonempty but input path is empty | `input path must not be empty.` |

Do not open either file or allocate a model on a usage error. For example,
`cinder run model.cnd` reports the second reason and exits `2`.

## Operational diagnostics

Use these one-line templates, ending with LF:

```text
cinder: model "<escaped-path>" [<STATUS>]: <detail>
cinder: input "<escaped-path>" [<STATUS>]: <detail>
cinder: inference [<STATUS>]: <detail>
cinder: output [IO]: Could not write prediction output.
```

Paths in errors are the full supplied arguments, not basenames. Escape paths
and the model display name byte by byte: printable ASCII `0x20..0x7e` passes
through except `\` becomes `\\` and `"` becomes `\"`; every other byte becomes
`\xHH` with two uppercase hexadecimal digits. This keeps unusual filenames on
one unambiguous physical line. Escaping affects display only, not file lookup.

`<STATUS>` is the corresponding library name with `CINDER_ERROR_` removed:
`INVALID_ARGUMENT`, `OVERFLOW`, `ALLOCATION`, `OUT_OF_BOUNDS`, `NONFINITE`,
`DIMENSION_MISMATCH`, `BUFFER_TOO_SMALL`, `IO`, `FORMAT`, `UNSUPPORTED`, or `LIMIT`.
An unrecognized failure code uses `UNKNOWN`. These diagnostic tags do not
introduce new library enums. Inference buffer allocation uses `ALLOCATION`.

For model/inference API failures, preserve the returned status and its bounded
`cinder_error_t.message`; use `Operation failed.` if the detail is empty. Escape
any control/non-ASCII bytes in a detail as `\xHH` so diagnostics stay on one line.
The template, stage, and tag are specified; existing library detail wording is
for people and may evolve. Do not append `strerror(errno)` to library failures:
the loader explicitly does not expose an `errno` contract.

Required input-reader and CLI-generated details are:

| Failure | Stage/tag | Exact detail template |
| --- | --- | --- |
| Input open | input / `IO` | `Could not open input file for reading.` |
| Seek to end | input / `IO` | `Could not seek to end of input file.` |
| Size query | input / `IO` | `Could not determine input file size.` |
| Length differs | input / `FORMAT` | `Expected N bytes (D float32 values); found A bytes.` |
| Seek to beginning | input / `IO` | `Could not seek to start of input file.` |
| Input buffer allocation | input / `ALLOCATION` | `Could not allocate input buffer.` |
| Size calculation cannot fit | input / `OVERFLOW` | `Input byte count is not representable.` |
| Read reports an I/O error | input / `IO` | `Input file read failed.` |
| Clean early EOF during reading | input / `FORMAT` | `Unexpected end of input file at byte B.` |
| Short read with neither error nor EOF | input / `IO` | `Incomplete input file read.` |
| NaN or either infinity | input / `NONFINITE` | `Input value at index I is not finite.` |
| Extra byte after the vector | input / `FORMAT` | `Trailing data after D float32 values.` |
| Final EOF probe has an error, or fails to confirm clean EOF | input / `IO` | `Final input EOF check failed.` |
| Input close after otherwise successful reading | input / `IO` | `Could not close input file.` |
| Output buffer allocation | inference / `ALLOCATION` | `Could not allocate output buffer.` |
| Output extent cannot fit | inference / `OVERFLOW` | `Output byte count is not representable.` |
| Loaded model unexpectedly reports a zero dimension | inference / `INVALID_ARGUMENT` | `Loaded model dimensions must be positive.` |
| A successful model call unexpectedly returns a non-finite score | inference / `NONFINITE` | `Output value at index I is not finite.` |

Substitute unsigned decimal values for `N`, `D`, `A`, `B`, and `I`: expected
bytes, expected element count, observed bytes, first missing byte offset, and
element index, respectively. Offsets and indices are zero-based. The input
reader must verify the host's binary32 representation before decoding; an
unsupported host uses input / `UNSUPPORTED` with detail
`Host float encoding is not IEEE binary32.` The model loader normally detects
this first, since model loading precedes input reading.

Examples of the diagnostic presentation:

```text
cinder: input "digit.bin" [FORMAT]: Expected 3136 bytes (784 float32 values); found 3135 bytes.
cinder: input "digit.bin" [NONFINITE]: Input value at index 17 is not finite.
cinder: model "future.cnd" [UNSUPPORTED]: At byte 16: Unsupported .cnd version (expected 1).
```

## Failure order, ownership, and cleanup

1. Validate invocation, then call `cinder_model_load` using an initially NULL
   model pointer. A model error wins over an input-file error; do not open input
   until the model has loaded successfully.
2. Obtain positive input/output dimensions through the public accessors. The
   model owns its layers, parameters, and two working buffers. CLI input and
   output buffers are separate caller-owned storage.
3. Open input once in `rb` mode. Check expected extent and measured length before
   allocation/decoding. Require complete reads, finite values, clean EOF, and
   successful close. Length rejection precedes value checks. If a short read
   also reports `ferror`, report `IO` rather than truncation.
4. Allocate the disjoint output buffer and call `cinder_model_run` once with
   exactly the model input length and enough output capacity. Propagate its
   first failure. Check finite final values and select the best three indices
   without modifying model parameters or inventing scores.
5. Keep the selected indices/scores available for display and release owned
   buffers/model. Only then emit the successful output and check its flush.

On every ordinary failure, release all acquired model and buffer allocations.
Every opened input stream gets exactly one close attempt; the loader owns its
own stream. Never retry a failed close or replace an earlier failure with a
cleanup error. A close failure with no earlier error prevents inference.
The library continues to return errors without printing; only the CLI writes
diagnostics. The process uses one model call and introduces no concurrent-run API.

## Implemented acceptance coverage

- Preserve exact `--version` output and empty stderr; reject every unsupported
  invocation with the defined status/reason/usage and no file access.
- Exercise paths with spaces, leading hyphens, quotes, control bytes, and symlinks;
  suffixes do not affect parsing. Confirm path escaping and basename display.
- Load the tiny fixture and MNIST fixtures through the CLI; use model APIs and
  saved references for numerical correctness. Displayed last digits may vary
  within the allowed runtime tolerance, so do not substitute exact Python text
  equality for numerical C/Python comparison.
- Test one/two/more-than-three outputs, negative scores, tied maxima and cutoff
  scores, signed zero, and non-Softmax models. Check sorting before formatting,
  exact separators/newlines, numeric locale, and absence of timing/percent signs.
- Reject short/long/empty inputs, NaNs/infinities, malformed models, unsupported
  versions, and resource limits. Verify zero stdout on failures before printing.
- Force input I/O/allocation/close, inference/output-allocation, and stdout-write
  failures. Verify precedence, exit status, diagnostics, and cleanup.

The original contract step changed documentation only. The subsequent
implementation adds `cli_input`, `cli_audit`, and `cli_process` tests beside
the version smoke test. Ordinary CTest uses committed fixtures without Python.
See [the demonstration report](../examples/mnist/DEMO.md#verification-of-the-cli-addition)
for measured local results, hosted status, and limits.
