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

## Scope

The gated program takes Host launch arguments and waits on a CKE on the
channel's IO die. The two-round program has one Host launch per L2 run and
reads each request from an independent GM mailbox. The Host exchanges the
data access tokens during setup. Orch records the AIV tasks' dependencies
through `CoreTaskArgs`. The resident wrapper takes positional arguments and
uses the existing TMR counter completion API.

The Host verifies the data and guards after the CCU stream drains. The
communicator, stream, and memory stay owned when completion is unconfirmed.
The experiment uses a private communicator and a bounded two-round program.
A production domain provider, program-key binding, L2 pool authorization,
and reuse of a completion slot inside a run are separate work.
