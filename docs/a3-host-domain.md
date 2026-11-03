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

`host_window` is local-L3-only; `host_windows` is L4-only and must name exactly
the participating nodes. The two arguments are mutually exclusive. Source node
and destination device rank are separate: the receiving L3 may map any source
Host backing onto one of its own domain members. Mapping at the L4 controller or
onto another L3 node's device is refused. Each consuming run must look up or
reuse the domain in its owner L4 callback before dispatch; a child lookup alone
does not enroll the owner's run, as with existing retained HBM domains. The native CPU pointer is usable only
in the owning L3 process; other nodes and L4 receive the same canonical Buffer
with `base=0`, not a remotely dereferenceable CPU pointer. Initialize or inspect
a Host table in its owning L3 callback, with device users excluded.

A3 owners use VMM Host physical memory and Fabric exports for both local and
remote consumers. Every chip reports its actual BareTgid before authorization.
The controller exchanges Host exports separately from the unchanged v2 rank
descriptor table, authorizes the consuming processes, imports HBM domain state,
and binds Host source metadata. A `map_view` request then imports the selected
Host backing through NoUC/Fabric and installs only its requested restricted
record in the target ImportRegistry. Payload movement uses the existing
MTE/PTO-SDMA interfaces and completion rules; no IPC HostRegister export or
extra copy service is involved. This route requires a supported HCCS fabric; it
does not introduce a RoCE fallback or infer reachability from an Ethernet link.

Each receiving L3 issues a disjoint subset of the existing 255 reference slots
per backing, sharing its issuer across its local chips. References are routing
indices, not authority; target records still validate access, descriptor,
context and liveness. Exhaustion fails explicitly without reusing stale refs.

Release confirms Host revocation/detach on **all** nodes before releasing any
owner. HBM domain release follows those acknowledgements; only after every node
confirms that stage are Host owners released. Partial prepare/import failures
use the same staged rollback. An unconfirmed stage retains owner cleanup debt;
a local success or child process exit is not a remote release acknowledgement.

The manual global Host acceptance case exercises peer reads, peer writes,
owner CPU readback and map/unmap restrictions using the existing vector-add
callable. Local simulation validates control and permission behavior, not
physical inter-server Fabric visibility or PID isolation. Run it explicitly
with `--manual only`; it adds no default scene-CI matrix. Dual-host A3 hardware
qualification, including unauthorized-PID rejection, remains required.

## Multi-node Host qualification

The manual global Host test reuses the existing vector-add kernel. Its local and
TCP variants cover two L3 nodes, peer Host reads and writes, CPU readback,
receiver selection, unmap/stale references and retained-domain cleanup. Both
simulator variants pass; this is not evidence of hardware HCCS reachability or
communication/computation overlap.

For a two-machine A3 run, install the same branch commit and PTO pin on the
controller and both nodes, with compatible CANN/driver versions. Reserve the
selected physical devices on both nodes through their normal device queues and
keep the reservations for the complete test. On each node, start the existing
worker daemon in that checkout's environment:

```bash
python -m simpler.remote_l3_worker --host 0.0.0.0 --port <port>
```

With two reachable endpoints and the two reserved physical device IDs in node
order, run from the controller:

```bash
SIMPLER_HOST_ENDPOINTS='<node0>:<port0>,<node1>:<port1>' \
  python -m pytest \
  'tests/st/worker/comm_domain/test_global_host_domain.py::test_global_host_peer_read_write_and_revoke[tcp]' \
  --platform a2a3 --device <device0>,<device1> --manual only -v --forked
```

The endpoints carry control messages; data access uses the A3 Fabric backend and
requires a real shared HCCS topology. The test does not substitute TCP payload
copies for device access. Without `SIMPLER_HOST_ENDPOINTS`, the TCP simulator
variant starts two loopback daemons and preserves their logs under its pytest
temporary directory. This test remains manual and adds no per-PR scene selection.
