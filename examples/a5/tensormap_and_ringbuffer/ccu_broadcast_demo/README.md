# A5 CCU Broadcast contract experiment

This experiment extends the two-rank Broadcast in
[Simpler #2786](https://github.com/hw-native-sys/simpler/pull/2786).
It uses CANN 9.3, two Ascend 950 devices with a UBC_CTP link, and the
Simpler A5 TensorMap/RingBuffer runtime.

## Run

Build and install Simpler in the worktree's own virtual environment. Source the
SDK and check the board architecture before reserving two permitted cards.
For example, when the local queue permits devices 0 and 1:

```bash
source /path/to/cann/set_env.sh
.claude/skills/onboard-arch-precheck/check.sh a5
task-submit --device 0,1 --max-time 120 --env PYTHONNOUSERSITE=1 \
  --env ASCEND_HOME_PATH --env LD_LIBRARY_PATH --env PATH \
  '.venv/bin/python examples/a5/tensormap_and_ringbuffer/ccu_broadcast_demo/test_ccu_broadcast_demo.py -d 0,1 --counts 64 --roots 0 --repeat 1 --memory worker-gm --reuse-runs 2'
```

- `--memory configured-gm` uses one `aclrtMallocWithCfg` allocation.
- `--memory plain-gm` uses separate `aclrtMalloc` allocations for source and destination.
- `--memory worker-gm` uses separate `ChipWorker.malloc` allocations.
- `--reuse-runs 2` retains one registered HCOMM handle across two L2 runs.
  The gated mode changes the input and length between runs. After the
  communicator is destroyed the native handle lookup must return `CCU_E_NOT_FOUND`.
- `--resident deferred` runs two requests in one L2 run. An AIV producer fills each
  source view. An AIV wrapper publishes the view addresses and byte count.
  After the CCU finishes that request it writes a GM completion record.
- `--resident early-control` runs the same graph and leaves out the completion
  registration. A bounded gate holds CCU until the consumer samples the
  sentinel or the gate reaches its deadline. Deferred mode must read the exact
  CCU result instead. Both modes use `--counts 64` and separate allocations.

## Reusable service

Use `--memory worker-gm --counts 64 --roots 0 --repeat 1 --service-rounds 1000`
to exercise one slot for 1000 requests in one L2 run. The count is an AIV
runtime scalar. The Host starts one registered CCU service per rank and run.
The AIV alternates two views and two lengths, verifies every result and its
neighboring guards, and checks that stale generations and submissions after
stop are rejected. Use `--service-rounds 0` to exercise stop without a request
and `--reuse-runs 2` to reuse the registration in a second L2 run.

The internal protocol allows one serialized AIV producer per execution context.
The producer flushes the arguments before publishing the command. CCU clears
an execute command and publishes its generation only after the operation and
peer synchronization finish. A stop command stays terminal and is acknowledged
with its own generation. The Host drains the service before reinitializing its
slot or releasing the communicator and buffers.

## L3 domain window probe

With two permitted devices, run `ccu_domain_probe.py -d 0,1` under the device
queue. The probe uses the public L3 domain and group submission APIs. A
fork-inherited test adapter prepares the private HCOMM session inside each
owner chip process, using the actual window addresses returned by Simpler.
It drives the producer, deferred CCU wrapper, and consumer in separate AIV
tasks, then drains and destroys the CCU session before releasing its window.
It also rejects a view from a second live domain on the same chip and repeats
the path after rebuilding the first domain under the same name.

## Scope

The gated program takes Host launch arguments and waits on a CKE on the
channel's IO die. The two-round program has one Host launch per L2 run and
reads each request from an independent GM mailbox. The Host exchanges the
data access tokens during setup. Orch records the AIV tasks' dependencies
through `CoreTaskArgs`. The resident wrapper takes positional arguments and
uses the existing TMR counter completion API.

The Host verifies the data and guards after the CCU stream drains. The
communicator, stream, and memory stay owned when completion is unconfirmed.
The experiment uses a private communicator. Callable-local ID resolution, a
production domain provider and TMR integration for the reusable slot are
separate work.
