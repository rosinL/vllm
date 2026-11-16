# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the per-request TTFT phase breakdown.

Covers:
* ``Request`` phase accumulator semantics (record / freeze / take-once).
* ``build_ttft_breakdown_log_line`` formatting and sum consistency.
* ``build_ttft_phase_breakdown`` response-model construction.
* ``OutputProcessor`` merge of engine-side ``phase_timings``.
"""

from typing import Any

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreOutput,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.output_processor import (
    OutputProcessor,
    build_ttft_breakdown_log_line,
)
from vllm.v1.metrics.stats import IterationStats, RequestStateStats
from vllm.v1.request import Request
from vllm.entrypoints.generate.base.serving import (
    build_per_request_timing_metrics,
    build_ttft_phase_breakdown,
)


def _make_request(request_id: str = "req-1") -> Request:
    request = Request.__new__(Request)
    request.request_id = request_id
    request.phase_timings = None
    request.load_kv_start_ts = None
    return request


class TestRequestPhaseTimings:
    def test_disabled_by_default(self):
        request = _make_request()
        assert request.phase_timings is None
        request.record_phase("hash", 0.1)
        request.record_chunk_durations(0.2, 0.3)
        assert request.take_phase_timings() is None

    def test_record_and_take_once(self):
        request = _make_request()
        request.phase_timings = {}
        request.record_phase("hash", 0.001)
        request.record_phase("lookup", 0.002)
        request.record_phase("alloc_slot", 0.003)
        request.record_phase("load_kv", 0.5)
        request.record_phase("sched_retries", 2)
        request.record_chunk_durations(0.01, 0.02)
        request.record_chunk_durations(0.03, 0.04)
        request.freeze_ttft_phases(0.005)
        # Frozen: further chunks are ignored.
        request.record_chunk_durations(9.0, 9.0)
        request.record_chunk_durations(9.0, 9.0)

        timings = request.take_phase_timings()
        assert timings is not None
        assert timings["hash"] == 0.001
        assert timings["lookup"] == 0.002
        assert timings["alloc_slot"] == 0.003
        assert timings["load_kv"] == 0.5
        assert timings["sampling"] == 0.005
        assert timings["chunks"] == [[0.01, 0.02], [0.03, 0.04]]
        assert "ttft_done" not in timings
        # Taken exactly once.
        assert request.take_phase_timings() is None

    def test_missing_chunks_when_none_forward(self):
        request = _make_request()
        request.phase_timings = {}
        request.record_chunk_durations(None, None)
        request.freeze_ttft_phases(None)
        timings = request.take_phase_timings()
        assert timings["chunks"] == [[0.0, 0.0]]
        assert "sampling" not in timings


def _make_stats(**kwargs: Any) -> RequestStateStats:
    stats = RequestStateStats()
    stats.queued_ts = 10.0
    stats.scheduled_ts = 12.0
    stats.first_token_ts = 13.0
    stats.last_token_ts = 13.5
    return stats


class TestLogLine:
    def test_full_line(self):
        stats = _make_stats()
        stats.frontend_phase_durations = {
            "req_resolve": 0.001,
            "tokenizer": 0.002,
            "add_request": 0.0005,
        }
        stats.engine_phase_durations = {
            "hash": 0.001,
            "lookup": 0.002,
            "alloc_slot": 0.001,
            "load_kv": 0.1,
            "sampling": 0.003,
            "sched_retries": 2,
        }
        stats.engine_chunk_durations = [[0.3, 0.05], [0.2, 0.02]]

        line = build_ttft_breakdown_log_line("chatcmpl-1", stats)
        assert line is not None
        # ttft = 13.0 - 12.0 = 1.0s; queue = 2.0s
        assert "req_id=chatcmpl-1" in line
        assert "ttft=1000.00 ms" in line
        assert "req-resolve=1.00 ms" in line
        assert "tokenizer=2.00 ms" in line
        assert "add_request=0.50 ms" in line
        assert "queue_time=2000.00 ms" in line
        assert "hash=1.00 ms" in line
        assert "lookup=2.00 ms" in line
        assert "alloc_slot=1.00 ms" in line
        assert "load_kv=100.00 ms" in line
        assert "step1_forward=300.00 ms" in line
        assert "step1_wait_for_save=50.00 ms" in line
        assert "step2_forward=200.00 ms" in line
        assert "step2_wait_for_save=20.00 ms" in line
        assert "sampling=3.00 ms" in line
        assert "sched_retries=2" in line
        # other = 1000 - (1 + 2 + 1 + 3 + 300 + 50 + 200 + 20) ms = 423 ms
        assert "other=423.00 ms" in line

    def test_sum_consistency(self):
        stats = _make_stats()
        stats.engine_phase_durations = {
            "hash": 0.4,
            "lookup": 0.1,
            "alloc_slot": 0.1,
            "sampling": 0.2,
        }
        stats.engine_chunk_durations = [[0.1, 0.0], [0.05, 0.05]]
        line = build_ttft_breakdown_log_line("r", stats)
        assert line is not None
        # in-window sum = 0.8 + 0.2 = 1.0s == ttft -> other = 0
        assert "other=0.00 ms" in line

    def test_other_clamped_non_negative(self):
        stats = _make_stats()
        # in-window sum (1.5s) exceeds ttft (1.0s): clamped at 0.
        stats.engine_phase_durations = {"hash": 1.5}
        line = build_ttft_breakdown_log_line("r", stats)
        assert "other=0.00 ms" in line

    def test_no_first_token_returns_none(self):
        stats = _make_stats()
        stats.first_token_ts = 0.0
        assert build_ttft_breakdown_log_line("r", stats) is None

    def test_empty_phases_minimal_line(self):
        stats = _make_stats()
        line = build_ttft_breakdown_log_line("r", stats)
        assert line is not None
        assert "load_kv" not in line
        assert "step1" not in line
        assert "other=1000.00 ms" in line


class TestResponseModel:
    def test_breakdown_populated(self):
        stats = _make_stats()
        stats.frontend_phase_durations = {
            "req_resolve": 0.001,
            "add_request": 0.0005,
        }
        stats.engine_phase_durations = {
            "hash": 0.001,
            "lookup": 0.002,
            "alloc_slot": 0.001,
            "load_kv": 0.1,
            "sampling": 0.003,
            "sched_retries": 1,
        }
        stats.engine_chunk_durations = [[0.3, 0.05]]

        breakdown = build_ttft_phase_breakdown(stats)
        assert breakdown is not None
        assert breakdown.req_resolve_ms == pytest.approx(1.0)
        assert breakdown.tokenizer_ms is None
        assert breakdown.add_request_ms == pytest.approx(0.5)
        assert breakdown.queue_time_ms == pytest.approx(2000.0)
        assert breakdown.hash_ms == pytest.approx(1.0)
        assert breakdown.lookup_ms == pytest.approx(2.0)
        assert breakdown.alloc_slot_ms == pytest.approx(1.0)
        assert breakdown.load_kv_ms == pytest.approx(100.0)
        assert breakdown.sampling_ms == pytest.approx(3.0)
        assert breakdown.sched_retries == 1
        assert breakdown.steps is not None
        assert len(breakdown.steps) == 1
        assert breakdown.steps[0].forward_ms == pytest.approx(300.0)
        assert breakdown.steps[0].wait_for_save_ms == pytest.approx(50.0)
        # other = 1000 - (1 + 2 + 1 + 3 + 300 + 50) = 643
        assert breakdown.other_ms == pytest.approx(643.0)

    def test_disabled_returns_none(self):
        assert build_ttft_phase_breakdown(None) is None
        assert build_ttft_phase_breakdown(RequestStateStats()) is None

    def test_timing_metrics_carries_breakdown(self):
        stats = _make_stats()
        stats.engine_phase_durations = {"hash": 0.001}
        stats.engine_chunk_durations = [[0.1, 0.0]]
        metrics = build_per_request_timing_metrics(stats, 5)
        assert metrics.time_to_first_token_ms == pytest.approx(1000.0)
        assert metrics.ttft_breakdown is not None
        assert metrics.ttft_breakdown.hash_ms == pytest.approx(1.0)

        # Without engine phases the breakdown is absent.
        metrics = build_per_request_timing_metrics(RequestStateStats(), 5)
        assert metrics.ttft_breakdown is None


class TestOutputProcessorMerge:
    def _make_engine_core_request(self) -> EngineCoreRequest:
        return EngineCoreRequest(
            request_id="r-1",
            external_req_id="ext-1",
            prompt_token_ids=[1, 2, 3],
            mm_features=None,
            sampling_params=SamplingParams(max_tokens=2),
            pooling_params=None,
            arrival_time=0.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
        )

    def test_merge_and_log_line(self, caplog):
        processor = OutputProcessor(
            None,
            log_stats=True,
            per_request_ttft_breakdown=True,
            pd_role="D",
        )
        processor.add_request(self._make_engine_core_request(), None)
        assert "r-1" in processor.request_states
        processor.record_frontend_phase("r-1", "req_resolve", 0.001)
        processor.record_frontend_phase("r-1", "tokenizer", 0.002)
        req_state = processor.request_states["r-1"]

        output = EngineCoreOutput(
            request_id="r-1",
            new_token_ids=[5],
            finish_reason=FinishReason.LENGTH,
            events=[
                EngineCoreEvent(EngineCoreEventType.QUEUED, 10.0),
                EngineCoreEvent(EngineCoreEventType.SCHEDULED, 12.0),
            ],
            phase_timings={
                "hash": 0.1,
                "sampling": 0.01,
                "chunks": [[0.2, 0.03]],
            },
        )
        iteration_stats = IterationStats()
        processor.process_outputs(
            [output], engine_core_timestamp=13.0, iteration_stats=iteration_stats
        )

        # Engine phases merged into the frontend stats.
        assert req_state.stats.engine_phase_durations == {
            "hash": 0.1,
            "sampling": 0.01,
        }
        assert req_state.stats.engine_chunk_durations == [[0.2, 0.03]]
        assert req_state.stats.frontend_phase_durations == {
            "req_resolve": 0.001,
            "tokenizer": 0.002,
        }
        # Engine-core event timestamps arrived.
        assert req_state.stats.queued_ts == 10.0
        assert req_state.stats.scheduled_ts == 12.0
        assert req_state.stats.first_token_ts == 13.0

        # The request finished; one breakdown log line was emitted.
        assert "r-1" not in processor.request_states
        records = [
            r for r in caplog.records if r.name == "vllm.per_request_phases"
        ]
        assert len(records) == 1
        message = records[0].getMessage()
        assert message.startswith("[D] ")
        assert "req_id=ext-1" in message
        assert "load_kv" not in message  # D-side request without async load
        assert "step1_forward=200.00 ms" in message
        assert "step1_wait_for_save=30.00 ms" in message
        # other = (13 - 12) - (0.1 + 0.01 + 0.2 + 0.03) = 0.66 s
        assert "other=660.00 ms" in message

    def test_breakdown_disabled_no_merge_no_log(self, caplog):
        processor = OutputProcessor(
            None,
            log_stats=True,
            per_request_ttft_breakdown=False,
            pd_role="",
        )
        processor.add_request(self._make_engine_core_request(), None)
        req_state = processor.request_states["r-1"]

        output = EngineCoreOutput(
            request_id="r-1",
            new_token_ids=[5],
            finish_reason=FinishReason.LENGTH,
            events=[
                EngineCoreEvent(EngineCoreEventType.QUEUED, 10.0),
                EngineCoreEvent(EngineCoreEventType.SCHEDULED, 12.0),
            ],
            phase_timings={"hash": 0.1, "chunks": [[0.2, 0.03]]},
        )
        processor.process_outputs(
            [output], engine_core_timestamp=13.0, iteration_stats=IterationStats()
        )
        # phase_timings still merged when present (engine ran with the flag),
        # but no log line because the frontend flag is off.
        assert req_state.stats.engine_phase_durations == {"hash": 0.1}
        records = [
            r for r in caplog.records if r.name == "vllm.per_request_phases"
        ]
        assert len(records) == 0
