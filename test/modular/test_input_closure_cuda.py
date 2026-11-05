"""Real CUDA reads before sealing, and descriptor rejection after sealing."""

import copy

import pytest
import torch

pytest.importorskip("mstar_rust")

from test_input_closure import _Case, _info

from mstar.communication.arena import ArenaShmCommunicationManager
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge
from mstar.utils.ipc_format import InputSignals

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _Recorder:
    def __init__(self):
        self.sent = []

    def send(self, peer, message):
        self.sent.append((peer, message))


@pytest.mark.parametrize("late_epoch", [-1, 0])
@pytest.mark.parametrize("native", [False, True], ids=["split", "native"])
def test_cuda_input_is_rejected_after_clear_seal(tmp_path, monkeypatch, late_epoch, native):
    monkeypatch.setenv("MSTAR_SHM_ARENA_SEGMENT_MB", "1")
    monkeypatch.setenv("MSTAR_SHM_ARENA_MAX_SEGMENTS", "4")
    case = _Case(tmp_path / "runtime")
    managers = []
    try:
        for entity in ["producer", "w0"]:
            manager = ArenaShmCommunicationManager(
                my_entity_id=entity, hostname="localhost", device="cuda:0",
                communicator=_Recorder(), shm_dir=str(tmp_path / "shm"),
            )
            managers.append(manager)
            cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[], shard_dim={})
            cfg.setup({})
            manager.register_request(case.rid, cfg)
            manager.rid_to_str = lambda rid: "r"
        producer, receiver = managers
        case.worker.tensor_manager = receiver
        expected = torch.arange(1024, dtype=torch.float32)
        edges = []
        for name in ["initial", "late"]:
            [info] = producer.store_and_return_tensor_info(
                case.rid, {name: [expected.cuda()]},
            )[name]
            producer.increment_ref(info.uuid)
            producer.register_for_send(case.rid, [info])
            assert info.shm_segment is not None
            edges.append(GraphEdge(
                name="input", next_node="n0", tensor_info=[info],
                _total_fanin=1, _shard_dim=None,
            ))

        for future in receiver.start_read_tensors(case.rid, [edges[0]]):
            future.result(timeout=10)
        assert receiver.get_ready_tensors()
        torch.testing.assert_close(receiver.get_tensor(edges[0].tensor_info[0].uuid).cpu(), expected)
        assert not receiver.has_inflight_reads(case.rid)
        report = case.finish(native=native)
        assert not report.has_inflight_reads

        reads = []
        start_read = receiver.start_read_tensors

        def observed_read(rid, inputs, **kwargs):
            reads.extend(inputs)
            return start_read(rid, inputs, **kwargs)

        monkeypatch.setattr(receiver, "start_read_tensors", observed_read)
        case.worker._process_new_inputs(InputSignals(
            request_id="r", inputs=[copy.deepcopy(edges[1])],
            request_info=_info(late_epoch), partition_name="default",
        ))
        assert reads == [], "late descriptor initiated a real CUDA read after the final clear report"
        assert report.closed_input_epoch == 0
        assert not receiver.tensor_store.check_uuid_presence(edges[1].tensor_info[0].uuid)

        case.worker._process_new_inputs(InputSignals(
            request_id="r", inputs=[copy.deepcopy(edges[1])],
            request_info=_info(1), partition_name="default",
        ))
        assert len(reads) == 1, "the next authorized epoch must still accept input"
        torch.cuda.synchronize()
        for pending in receiver.pending:
            if pending.future is not None:
                pending.future.result()
        assert receiver.get_ready_tensors()
        torch.testing.assert_close(receiver.get_tensor(edges[1].tensor_info[0].uuid).cpu(), expected)
    finally:
        torch.cuda.synchronize()
        for manager in managers:
            manager.force_cleanup_request(case.rid)
            manager.close()
