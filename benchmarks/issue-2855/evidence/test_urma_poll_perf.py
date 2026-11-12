"""A5 spot check using the production URMA scene and golden."""

from time import perf_counter_ns
from unittest.mock import patch

from simpler_setup import scene_test
from tests.st.a5.tensormap_and_ringbuffer.urma_deferred_completion.test_urma_deferred_completion import (
    BENCHMARK_PUT_MODE,
    BENCHMARK_WINDOW_SLOTS,
    DTYPE_NBYTES,
    _UrmaRegression,
    _case,
)


@scene_test(level=3, runtime="tensormap_and_ringbuffer")
class TestUrmaPollPerf(_UrmaRegression):
    CASES = [
        case
        for kib, batch in ((4, 16), (32, 16), (128, 4), (224, 4), (256, 4))
        for size in (0, batch)
        for case in _case(
            f"put_{kib}k_{f'batch{size}' if size else 'immediate'}",
            mode=BENCHMARK_PUT_MODE,
            batch_size=size,
            elems=BENCHMARK_WINDOW_SLOTS * kib * 1024 // DTYPE_NBYTES,
            manual=True,
        )
    ]

    def _run_and_validate(self, worker, callable_obj, case, **kwargs):
        original_run = type(worker).run

        def timed_run(worker_instance, *args, **kwargs):
            start = perf_counter_ns()
            result = original_run(worker_instance, *args, **kwargs)
            duration = perf_counter_ns() - start
            print(f"URMA_HOST_L3_NS {duration}", flush=True)
            return result

        with patch.object(type(worker), "run", timed_run):
            return super()._run_and_validate(worker, callable_obj, case, **kwargs)
