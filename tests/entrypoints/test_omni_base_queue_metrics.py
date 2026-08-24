from vllm_omni.engine.messages import DiffusionQueueStatsMessage
from vllm_omni.entrypoints.omni_base import OmniBase


class _PrometheusRecorder:
    def __init__(self) -> None:
        self.waiting: int | None = None
        self.running: int | None = None

    def set_waiting(self, value: int) -> None:
        self.waiting = value

    def set_running(self, value: int) -> None:
        self.running = value


def test_diffusion_scheduler_backlog_is_pipeline_waiting() -> None:
    omni = object.__new__(OmniBase)
    omni.request_states = {f"req-{index}": object() for index in range(10)}
    omni.prom_metrics = prom = _PrometheusRecorder()

    omni._process_diffusion_queue_stats_message(
        DiffusionQueueStatsMessage(stage_id=0, replica_id=0, waiting=9, running=1)
    )

    assert prom.waiting == 9
    assert prom.running == 1


def test_diffusion_queue_stats_aggregate_replicas_without_overcounting() -> None:
    omni = object.__new__(OmniBase)
    omni.request_states = {f"req-{index}": object() for index in range(3)}
    omni.prom_metrics = prom = _PrometheusRecorder()

    omni._process_diffusion_queue_stats_message(DiffusionQueueStatsMessage(stage_id=0, replica_id=0, waiting=2, running=1))
    omni._process_diffusion_queue_stats_message(DiffusionQueueStatsMessage(stage_id=0, replica_id=1, waiting=2, running=0))

    assert prom.waiting == 3
    assert prom.running == 0
