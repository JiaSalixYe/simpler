# Direct A3 Host buffers

A direct L2 Worker can allocate a Host backing and explicitly select a device
mapping. Buffer remains HOST with the same canonical identity; mapping returns a
new Tensor. Transfer policy belongs to each TaskArgs argument.

```python
buffer = worker.create_buffer(4096, host_window=HostWindowSpec(4096))
native = buffer.tensor((1024,), DataType.FLOAT32)
mapped = worker.map_view(native, access=AccessMode.READ)
# Use mapped with TensorTransfer.NONE, or native with explicit H2D.
# After all accepted uses finish:
worker.unmap_view(mapped)
worker.release_buffer(buffer)
```

`Buffer.base` is the creator's CPU pointer; direct CPU reads and writes require
caller synchronization. `Buffer.close()` closes logical admission but does not
replace `Worker.release_buffer()` for the owning allocation.

## Mapping and ownership

A3 VMM/Fabric backing, CPU owner view, export and PID authorization reside in the
A3 platform runtime DSO. The common bridge preserves module lifetime while an
owner remains retained. The extension does not compile the A3 owner strategy.
Simulation uses its existing shared-memory backend. A5 is unsupported here.

The target ImportRegistry validates the canonical descriptor, selected grant,
execution context and liveness through existing adapter/materialization paths.
Mapping references are local opaque indices, never reusable within the backing
lifetime; a copied integer supplies no authority. References are limited to 255
issued records per backing. The mapping grant covers the backing extent; Tensor
geometry bounds each argument. Callable signatures determine required access.
Native and mapped arguments resolve independently even when identities match.

`unmap_view` rejects in-flight use before revocation. Once cleanup starts, failed
cleanup remains closed to new use until retry confirms completion. Removing the
last active record detaches the physical device view; remapping issues a fresh
reference. Member views must be confirmed gone before the CPU owner is freed.
Unknown native completion retains the backing and cleanup debt, rather than
assuming that a failed call means no device user remains.

Direct L2 internally reuses a one-member communicator and 64-byte control window;
Host is an attachment, not an additional rank. This control allocation does not
stage the Host table. Public objects remain Buffer, Tensor and TaskArgs.

## Scope

This is A3 single-node resource and mapping support, qualified through direct L2
HBG/TMR MTE tests. It does not claim A5, cross-node access, asynchronous fetch
progress or communication/computation overlap required by the full #2657 goal.
See [Buffer ABI](buffer-abi.md) for the fixed-size Tensor mapping fields and
same-build native boundary.

The L2 acceptance suite covers TMR native/H2D plus mapped reads with unmap/remap,
HBG native HOST/NONE plus mapped reads, HBG WRITE, and TMR READWRITE. The full
transfer/runtime Cartesian product is intentionally absent. Per-PR scene CI
keeps the TMR mixed-view/unmap path and HBG native HOST/NONE consumer path.
WRITE/READWRITE qualification runs explicitly with `--manual only` (or with
`--manual include` for the full acceptance suite). Runtime-independent grant,
context, stale-reference and lifetime failures remain in unit-test CI.
