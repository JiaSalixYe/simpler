# A3 Host communication buffers

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

Local A3 resource and mapping support is qualified through direct L2 HBG/TMR
MTE tests. The cross-node extension below uses the same contract and requires
separate hardware qualification. Neither path claims A5, asynchronous fetch
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

## L3 multi-worker domains

L3 reuses the same backing, restricted mapping records and platform backend.
It adds a Host attachment to an existing domain without adding a rank:

```python
domain = orch.get_or_allocate_domain(
    name="host-table", members=[0, 1], window_size=65536,
    buffers=[CommBufferSpec("output", "float32", 16384, 65536)],
    host_window=HostWindowSpec(nbytes=65536, numa_id=0),
    retain_after_run=True,
)
orch.copy_to_host_domain(domain, data)
native = domain.host().tensor((16384,), DataType.FLOAT32)
mapped = domain.map_view(native, domain_rank=0, access=AccessMode.READ)
args = TaskArgs()
args.add_tensor(mapped, TensorArgType.INPUT, transfer=TensorTransfer.NONE)
# Submit through orch.submit_next_level(..., worker=0).
```

`members` contains chip-worker indices; `domain_rank` selects a position in that
list. Each child reports its actual process identity before owner authorization.
Host exports use the existing v2 descriptor envelope outside the rank table;
Python and native artifacts must come from the same build.

Host staging accesses the local CPU backing. The ranks accepted by
`copy_to_domain` and `copy_from_domain` remain NPU ranks. L3 native Host parameters
are not a device-consumption entry. Retained domains support reuse across runs.

### Ordering and collective cleanup

In an orchestration callback, Host staging and map/unmap must precede its first
submission. Outside callbacks, accepted runs must have drained. These operations
reuse Worker admission, control reservation and endpoint locks; invalid ordering
fails before data is changed. Raw CPU writes bypass these checks.

Domain release waits for accepted uses, confirms every child view is gone, then
releases the owner. Unknown completion or an unconfirmed child result retains the
backing. Local detach success does not prove whole-domain release. Native partial
cleanup retains state without repeating a completed barrier; Python conservatively
replays an unconfirmed collective failure. Worker.close still tears down children
and reports unresolved cleanup debt. Destructors cannot substitute for release
confirmation.

L3 acceptance covers HBG/TMR retained BF16 selected-row reads and accepted-use
release, TMR MTE Host writes, and bidirectional PTO SDMA completion/consumer
visibility. Exact-head validation is required;
historical results are background evidence. Unauthorized-PID negative hardware
qualification, Engram layout and asynchronous latency-hiding measurements remain
separate work. Request completion does not permit overwriting output still held
by a consumer. This feature does not close #2460 or #2657.

## CI and qualification

Per-PR L3 scenes add TMR two-worker write/reuse and HBG accepted-use release.
TMR release qualification and HBG/TMR BF16 application fetches run explicitly
with `--manual only`. Hardware-only SDMA directions retain their isolated CI
lane. Full qualification uses the same module with `--manual include`; this
selection does not replace onboard evidence or establish latency hiding.

## Cross-node HCCS Host backing

An L4 domain can request one Host backing on each participating L3 node:

```python
domain = orch.get_or_allocate_domain(
    name="shared-tables", members=[(0, 0), (1, 0)], window_size=65536,
    buffers=[CommBufferSpec("output", "float32", 16384, 65536)],
    host_windows={0: HostWindowSpec(65536), 1: HostWindowSpec(65536)},
    retain_after_run=True,
)
# Pass domain.ref through the existing L3 callable route. In the receiving L3:
domain = orch.get_domain(domain_ref)
source = domain.host(1).tensor((16384,), DataType.FLOAT32)
mapped = domain.map_view(source, domain_rank=0, access=AccessMode.READ)
# Submit mapped with transfer NONE to that member, then retire its accepted use.
# In a later callback or after the run drains:
domain.unmap_view(mapped)
```

`host_window` is local-L3-only; L4 `host_windows` must name exactly its participating
nodes. They are mutually exclusive. The source node and receiving device rank
are separate: `map_view` must run on the L3 owning the target member. Only the
source L3 has a usable CPU `Buffer.base`; other L3 nodes and L4 get `base=0`.
Initialize/update tables in the owner L3 callback with device users excluded.
Each consuming run must look up or reuse the domain in its owner L4 callback;
a child lookup alone does not enroll that run.

A3 reuses the platform DSO's VMM/Fabric owner and NoUC import. Actual chip
BareTgids authorize the immutable Host export table before IMPORT; Host exports
remain separate from the unchanged v2 rank descriptors. First MAP carries the
source descriptor and establishes its restricted target record, without a
separate metadata BIND fanout. Later MAPs must match that source. Transfers
reuse existing MTE/PTO-SDMA and completion interfaces. A5 and RoCE fallback are
not implemented here; this route requires a supported HCCS fabric.

Receiving L3s partition the existing 255 non-reusable reference slots per backing.
References provide no authority; target records still validate descriptor,
grant, context and liveness. Exhaustion fails explicitly.

Existing domain release revokes target records and retires native Host imports
before HBM. Only after every node acknowledges that release are Host owners
closed. Partial creation uses the same rollback. Unconfirmed results
retain cleanup debt; local child exit cannot substitute for remote confirmation.
This does not establish asynchronous latency hiding or complete #2657.

### Dual-host qualification

Install the same commit and PTO pin on controller and both A3 nodes, with
compatible CANN/drivers. Keep physical-device queue reservations on both nodes
for the complete test. Start the existing daemon in each checkout's environment:

```bash
python -m simpler.remote_l3_worker --host 0.0.0.0 --port <port>
```

From the controller, name endpoints and reserved physical device IDs in node order:

```bash
SIMPLER_HOST_ENDPOINTS='<node0>:<port0>,<node1>:<port1>' \
  python -m pytest \
  'tests/st/worker/comm_domain/test_global_host_domain.py::test_global_host_peer_read_write_and_revoke[tcp]' \
  --platform a2a3 --device <device0>,<device1> --manual only -v --forked
```

TCP carries control; peer data uses Fabric/HCCS. Without external endpoints, the
`a2a3sim` case starts two loopback daemons and saves logs in pytest's temporary
directory. The manual local/TCP cases reuse vector-add for peer reads/writes,
CPU readback, receiver selection, unmap/stale references and retained cleanup;
they add no default scene-CI matrix. Simulation cannot qualify real inter-server
visibility or unauthorized-PID rejection; those require dual-host hardware.
