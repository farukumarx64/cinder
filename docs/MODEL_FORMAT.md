# Cinder model format — version 1

**Contract defined October 23, 2026.** This document specifies the `.cnd` file
format for Cinder v0.1.0. [Python validation and binary export](EXPORTER.md) are
implemented, as is the [C loader API](LOADER.md), with initial successful-load
and malformed-file tests. The format version is
the integer `1`, independent of the runtime's `0.1.0-dev` version string.

This is the authoritative byte layout. Writers must produce it and readers
must validate it. It describes one fixed-shape, nonempty sequential model for
one input vector, using Dense, ReLU, and Softmax. Its execution and ownership
rules follow the [model contract](MODEL.md) and [layer contract](LAYER.md).

## Encoding and compatibility

- A byte is an 8-bit octet. All offsets and lengths below count bytes unless
  explicitly described as element counts. Offsets start at zero.
- `u32le` is an unsigned 32-bit integer occupying exactly four bytes, least
  significant byte first. Dimensions and layer counts are element counts.
- `f32le` is one IEEE 754 binary32 value occupying exactly four bytes, with its
  32-bit representation in little-endian byte order. For example, `1.0` is
  `00 00 80 3f`, `-1.0` is `00 00 80 bf`, and `0.5` is `00 00 00 3f`.
- All parameters must be finite. Both signed zeros, subnormals, and finite
  extreme values are permitted. Preserve their binary32 encodings when writing
  and decoding; reject every NaN and both infinities. A finite model can still
  produce arithmetic overflow during inference, handled by the existing API.
- There is no alignment padding, native-endian encoding, implicit activation,
  compression, checksum, string, timestamp, or optional extension in version 1.
  Every byte belongs to a defined header or parameter payload.
- Never serialize native C structs, `size_t`, pointers, or enum objects. Decode
  fixed-width integers explicitly; convert float representations without
  unaligned pointer casts or aliasing violations. Raw `fread` into runtime
  objects is not an implementation of this contract.
- A C implementation must verify that its host can represent the required
  8-bit bytes and IEEE binary32 `float` values. `sizeof(float) == 4` alone is
  insufficient. Unsupported hosts must be rejected explicitly; host byte order
  must never change the file layout. Initial implementation targets remain
  the verified macOS/Linux toolchains.

Only version `1` is accepted. Reject zero and every other version; do not guess
an alternative layout or partially load a newer version. New fields, operators,
or changed meanings require a new version with a separately defined contract.
The `.cnd` suffix is a naming convention; the bytes determine validity.

## File header

The file begins with exactly **16 bytes**:

| Absolute offset | Width | Type | Field | Required value or meaning |
| --- | --- | --- | --- | --- |
| 0 | 4 | Raw bytes | `magic` | ASCII `CNDR`: `43 4e 44 52`, with no NUL terminator. |
| 4 | 4 | `u32le` | `version` | Exactly `1`. |
| 8 | 4 | `u32le` | `input_size` | Positive external input-vector length. |
| 12 | 4 | `u32le` | `layer_count` | Positive number of layer records that follow. |

The first record starts at offset 16. There is no header-size field, model-output
field, total-parameter field, or index table. The output size is the last
record's `output_size`; parameter totals are derived from validated records.
An empty sequence is rejected, consistent with `cinder_model_create`.

## Layer records

Records occur in execution order. Each contains a **16-byte record header**
immediately followed by `payload_bytes` bytes of data:

| Offset from record start | Width | Type | Field | Meaning |
| --- | --- | --- | --- | --- |
| 0 | 4 | `u32le` | `type_id` | File-format layer identifier from the table below. |
| 4 | 4 | `u32le` | `input_size` | Number of floats consumed. |
| 8 | 4 | `u32le` | `output_size` | Number of floats produced. |
| 12 | 4 | `u32le` | `payload_bytes` | Parameter bytes only; excludes the record header. |

The next record begins at `record_start + 16 + payload_bytes`, using checked
arithmetic. Length is redundant by design: it must equal the size independently
derived from the type and dimensions. It is never permission to skip unknown
operators or to allocate an unchecked amount of memory.

| File `type_id` | Layer | Required dimensions | Exact `payload_bytes` |
| --- | --- | --- | --- |
| `1` | Dense | Positive input and output sizes; either may be larger. | `4 * (input_size * output_size + output_size)` |
| `2` | ReLU | `input_size == output_size`, both positive. | `0` |
| `3` | Softmax | `input_size == output_size`, both positive. | `0` |

Zero and every unlisted ID are invalid in version 1. These are stable **file
IDs**, independent of the values or storage width of `cinder_layer_type_t`.
The loader must explicitly map `1` to `CINDER_LAYER_DENSE`, `2` to
`CINDER_LAYER_RELU`, and `3` to `CINDER_LAYER_SOFTMAX`; numerical coincidence
with today's runtime enum does not make the enum a serialization contract.

ReLU and Softmax have no parameter payload. Softmax operates across the entire
single feature vector using the existing stable operation. No axis field is
stored. There is no batch dimension. Softmax may appear wherever a compatible
layer is allowed, and the last layer need not be Softmax. One-layer and
activation-only models are valid; no layer is inserted or removed implicitly.

## Dense payload and connections

For a Dense record with input size `I` and output size `O`, the payload contains:

1. Exactly `O * I` weights, as `f32le`, in row-major `[O][I]` order.
2. Exactly `O` biases, as `f32le`, in output order.

There are no per-array length prefixes. Relative to the payload start,
`W[i, j]` is at byte offset `4 * (i * I + j)`, and `b[i]` is at
`4 * (O * I + i)`. Each weight row supplies one output:

```text
y[i] = sum(W[i, j] * x[j], j = 0 ... I - 1) + b[i]
```

Bias is mandatory, including explicit zero values when appropriate. There is
no transposition, broadcasting, scale factor, or fused activation. The runtime's
bias tensor may have shape `[O][1]`; the file stores its `O` values contiguously.

The header's input size must equal the first record's input size. Every later
record's input size must equal the preceding record's output size. Activations
must additionally preserve length. Validate all connections before publishing
an executable model; do not truncate, pad, reshape, or repair mismatches.

## Resource limits and size arithmetic

Version 1 uses the following **inclusive bounds** in both exporter and loader:

| Quantity | Minimum | Maximum |
| --- | --- | --- |
| Layer count | 1 | 1,024 |
| Header input size and every record input/output size | 1 | 65,536 floats |
| Sum of all Dense parameter payloads, including biases | 0 | 67,108,864 bytes (64 MiB) |
| Whole file size | 32 bytes | 67,125,264 bytes (`16 + 16 * 1024 + 67,108,864`) |

Zero total parameter bytes is valid for activation-only models. Each Dense
payload is positive and must fit within the remaining total-parameter budget.
The whole-file maximum is an upper bound; it does not make arbitrary padding
valid. The actual length must equal `16 + 16 * layer_count + sum(payload_bytes)`.
The minimum is a header plus one activation record.

These bounds keep work and storage limited for a small runtime while admitting
the planned `784 -> 128 -> 10` MNIST network: 101,770 parameters / 407,080 bytes.
The largest vector bound requires at most 262,144 bytes per runtime working
buffer, or 524,288 bytes for both. Parameter-copy staging, layer descriptors,
pointer arrays, and allocator overhead are additional memory; 64 MiB is a
parameter limit, not a promise about peak process memory or execution time.
The in-memory constructors retain their existing contracts; these new limits
apply to `.cnd` export and loading.

Validation must precede arithmetic, allocation, indexing, and reads that depend
on the corresponding fields:

1. Validate the complete fixed header, supported version, and header bounds.
   Reject a known file size above the cap and insufficient bytes even for the
   declared record headers before allocating a layer list.
2. For each complete record header, validate its ID, dimensions, and connection.
   Compute weight count, bias count, and expected payload bytes in checked
   arithmetic. In C, widen operands before multiplying; a wider destination
   does not repair multiplication that has already overflowed.
3. Compare the computed payload size with `payload_bytes`. Check its field
   representability and cumulative parameter limit. Use subtraction against
   the remaining budget, or checked addition; never add first and check later.
4. Verify that the record's byte range exists in the input before allocating
   its parameter storage. Check every conversion to `size_t`, every native
   allocation size, pointer-list size, workspace size, and file-offset advance.
   The two allowed dimensions can still produce an invalid oversized matrix.
5. Decode and validate every parameter, perform exact reads, and reject short
   reads or I/O failures. A prior file-size check does not replace checking
   actual reads. No allocation may be based solely on an untrusted length.
6. After all declared records, require exact end of file, then finish model
   construction and publish success. A read error is not a clean EOF.

Bounded staging or checking a seekable file's remaining length may establish
byte availability; this contract does not require reading the entire file into
one allocation. Allocation can still fail within the limits and must return
an error with complete cleanup. Limits do not make all structurally valid
networks numerically safe for every input.

## End of file, names, and failure behavior

EOF must occur immediately after the final declared record's payload. Reject
**any extra byte**, including zero padding, whitespace, another model, or an
`END` marker. There is no footer. Missing headers or payload bytes are errors;
a smaller declared layer count cannot make leftover records acceptable.

No model name, class labels, training metadata, preprocessing instructions,
paths, or reference outputs are embedded. Future `cinder info` displays the
basename of the supplied path, including its extension: `/models/tiny.cnd`
is displayed as `Model: tiny.cnd`. Renaming a file changes that display label,
not its model semantics. Example documentation supplies human descriptions and
class meanings separately. Writers must not append such metadata to the file.

The loader must reject malformed or unsupported files, dimension
mismatches, resource-limit violations, non-finite parameters, allocation
failures, and I/O failures with useful diagnostics. It must release all partial
layers, parameter staging, model storage, and owned file resources, and leave
the caller's model output slot NULL. Successful loading yields independently
owned runtime layers and parameters; inference must not depend on an open file
or borrowed file bytes. Construct through the existing ownership-transfer path.

The [C loader contract](LOADER.md) now defines the public declaration, status
mapping, validation sequence, seekable binary file handling, and cleanup.
The function and file-related status codes are declared in the public headers;
[`src/loader.c`](../src/loader.c) implements the parser and construction path.
No invalid file may be accepted regardless of which of its violations is
reported first. File validation does not execute the network or weaken the
model's inference, output-preservation, or concurrency rules.

There is no checksum or authenticity mechanism in version 1. Structural checks
cannot detect a bit flip that changes one finite weight into another finite
weight. The saved fixtures and reference comparisons validate known models;
they do not provide general file-integrity detection.

## Input-file and preprocessing contract

The future `cinder run` and `cinder bench` commands take a separate input file:

- Exactly `model_input_size` consecutive `f32le` values, totaling
  `4 * model_input_size` bytes, in feature-index order.
- No header, count prefix, padding, multiple samples, or trailing data.
- Every value must be finite; signed zeros and subnormals remain permitted.
  Reject missing/extra bytes and I/O errors before inference.
- The caller supplies already-preprocessed values. The runtime does not infer
  normalization from the model filename or perform image decoding.

For MNIST, flatten a `28 x 28` image in row-major order: feature
`row * 28 + column` corresponds to that pixel. Convert the original `0..255`
pixel to float32 and divide by float32 `255.0`, producing `[0, 1]`. Training,
Python reference inference, and sample generation must use this same procedure.
The sample file is exactly 784 floats / 3,136 bytes. The general input format
allows any finite values; the `[0, 1]` restriction belongs to this MNIST example.
Example/export tooling will generate samples; preprocessing is not stored in
the `.cnd` model.

## Worked tiny network

This uses the existing [`numpy_tiny_network` parameters](../tests/fixtures/numpy_fixtures.h)
and the `worked` case, generated by
[`tiny_network()`](../tests/generate_numpy_fixtures.py). Reuse all ten saved
inputs and intermediate references for the later exporter/loader tests.

```text
input(2) -> Dense(2 -> 4) -> ReLU(4) -> Dense(4 -> 2) -> Softmax(2)

W1 = [[ 1,    2  ],     b1 = [0.5, -1, 2, 0]
      [-1,    0  ],
      [ 0.5, -0.5],
      [ 3,    1  ]]

W2 = [[1, 0, -1,   0.5],     b2 = [0, 0.5]
      [0, 1,  0.5, -1  ]]
```

The header is `magic = CNDR`, `version = 1`, `input_size = 2`, and
`layer_count = 4`. The derived output size is 2. Records and payload locations
are as follows; every offset in this table is absolute and decimal:

| Record index | Header start | Type ID | Input | Output | Payload bytes | Payload start | Next offset |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | 1 (Dense) | 2 | 4 | 48 | 32 | 80 |
| 1 | 80 | 2 (ReLU) | 4 | 4 | 0 | 96 | 96 |
| 2 | 96 | 1 (Dense) | 4 | 2 | 40 | 112 | 152 |
| 3 | 152 | 3 (Softmax) | 2 | 2 | 0 | 168 | 168 (EOF) |

For zero-length payloads, the listed start is only a boundary; no payload byte
exists there. The Dense payloads break down as:

| Array | Shape | Float count | First byte offset | Byte count |
| --- | --- | --- | --- | --- |
| `W1` | `4 x 2` | 8 | 32 | 32 |
| `b1` | `4 x 1` | 4 | 64 | 16 |
| `W2` | `2 x 4` | 8 | 112 | 32 |
| `b2` | `2 x 1` | 2 | 144 | 8 |

There are 22 parameters / 88 payload bytes. Total file length is exactly
`16 + 4 * 16 + 88 = 168` bytes. These are **all bytes of the example model**;
the labels before the colons are hexadecimal offsets, not stored data:

```text
0000: 43 4e 44 52 01 00 00 00 02 00 00 00 04 00 00 00
0010: 01 00 00 00 02 00 00 00 04 00 00 00 30 00 00 00
0020: 00 00 80 3f 00 00 00 40 00 00 80 bf 00 00 00 00
0030: 00 00 00 3f 00 00 00 bf 00 00 40 40 00 00 80 3f
0040: 00 00 00 3f 00 00 80 bf 00 00 00 40 00 00 00 00
0050: 02 00 00 00 04 00 00 00 04 00 00 00 00 00 00 00
0060: 01 00 00 00 04 00 00 00 02 00 00 00 28 00 00 00
0070: 00 00 80 3f 00 00 00 00 00 00 80 bf 00 00 00 3f
0080: 00 00 00 00 00 00 80 3f 00 00 00 3f 00 00 80 bf
0090: 00 00 00 00 00 00 00 3f 03 00 00 00 02 00 00 00
00a0: 02 00 00 00 00 00 00 00
```

The separate input `[2, -1]` occupies eight bytes:

```text
00 00 00 40 00 00 80 bf
```

It must produce the following stage values, with approximate decimals shown
for the saved NumPy Softmax reference:

| Stage | Values |
| --- | --- |
| First Dense | `[0.5, -3, 3.5, 5]` |
| ReLU | `[0.5, 0, 3.5, 5]` |
| Second Dense / logits | `[-0.5, -2.75]` |
| Softmax | `[0.9046505094, 0.0953494608]` |

For example, the first logit is `0.5 - 3.5 + 0.5 * 5 = -0.5`, and the
second is `0 + 0.5 * 3.5 - 5 + 0.5 = -2.75`. Output comparison first requires
matching shapes and finite values, then uses the existing rule for every value:

```text
abs(actual - reference) <= 1e-5 + 1e-5 * abs(reference)
```

The bytes, parameters, and inputs must round-trip exactly; inference results
use this numerical tolerance. Sample inputs and expected outputs are separate
fixtures, never appended to the model. See [NumPy verification](NUMPY_TESTS.md)
for the remaining cases and comparison diagnostics.

## Verification and next steps

The documented example was independently packed and decoded using Python's
standard-library `struct` module with explicit little-endian formats. Its 168
bytes, record/payload offsets, 22 parameters, eight input bytes, and intermediate
results were checked against the existing saved fixtures. The resource-limit
arithmetic and planned MNIST parameter count were also checked. This validates
the specification's example, not an exporter/loader implementation.

On October 24, 2026, the binary exporter produced these exact 168 bytes in
[`tests/fixtures/tiny/tiny.cnd`](../tests/fixtures/tiny/tiny.cnd). Independent
inspection checked every header, offset, parameter, and EOF. Repeated regeneration
matched the saved model, ten input files, and all expected stages byte for byte.
See [the exporter guide](EXPORTER.md) for the API, file-failure policy, and tests.

The C loader now loads the saved tiny network and compares final probabilities
against the saved NumPy references. Malformed-file, exact-limit, and allocation/I/O
failure cleanup tests now pass locally. Every saved stage also passes through
the [complete file round trip](ROUND_TRIP.md), with committed files and fresh
Python exports. [Loaded-model lifecycle checks](M3_VERIFICATION.md), local
sanitizers, and native leak checks also pass. Final macOS/Linux CI verification
of these additions remains required before M3 closes.
Ordinary C tests must consume saved fixtures without requiring Python.
