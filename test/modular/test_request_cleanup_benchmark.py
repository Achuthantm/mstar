"""The later GPU sweep must stay reproducible and reject failed runs."""

import json
import shlex
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from benchmark.transport import request_cleanup_gpu as sweep


@pytest.mark.parametrize("model, cases", [("qwen", 6), ("orpheus", 3), ("omni", 3)])
def test_dry_run_does_not_probe_hardware_or_run_clients(tmp_path, monkeypatch, capsys, model, cases):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run started a probe or client")

    monkeypatch.setattr(sweep, "urlopen", forbidden)
    monkeypatch.setattr(sweep.subprocess, "run", forbidden)
    monkeypatch.setattr(sweep.subprocess, "check_output", forbidden)
    out = tmp_path / "later"
    assert sweep.main(["--out", str(out), "--model", model, "--dry-run"]) == 0
    commands = [shlex.split(line) for line in capsys.readouterr().out.splitlines()]
    assert len(commands) == cases
    assert {arg for command in commands for arg in command if arg.startswith("CONC=")} == {
        "CONC=1", "CONC=32", "CONC=64",
    }
    if model == "qwen":
        assert all(command[-2:] == ["--dataset", "text"] for command in commands)
        assert {arg for command in commands for arg in command if arg.startswith("OUTMIN=")} == {
            "OUTMIN=16", "OUTMIN=512",
        }
    else:
        assert all(not any(arg.startswith("OUTMIN=") for arg in command) for command in commands)
        assert all("--ignore-eos" not in command for command in commands)
    assert not out.exists()


def test_completed_run_is_read(tmp_path):
    result = {
        "completed": 64, "failed": 0, "request_throughput": 2.0,
        "aggregate": {"max_concurrency": 32},
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result))
    assert sweep.read_result(path, expected=64, concurrency=32) == result


@pytest.mark.parametrize("changes", [
    {"failed": 1}, {"completed": 63}, {"request_throughput": 0},
    {"aggregate": {"max_concurrency": 1}},
])
def test_failed_or_mismatched_results_are_rejected(tmp_path, changes):
    result = {
        "completed": 64, "failed": 0, "request_throughput": 2.0,
        "aggregate": {"max_concurrency": 32},
    }
    result.update(changes)
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        sweep.read_result(path, expected=64, concurrency=32)


def test_run_directory_cannot_be_reused(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("existing output directory was not refused before probing")

    monkeypatch.setattr(sweep, "urlopen", forbidden)
    with pytest.raises(FileExistsError):
        sweep.main(["--out", str(tmp_path)])


def test_uncommitted_checkout_is_refused_before_starting_clients(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "urlopen", lambda *a, **k: nullcontext(SimpleNamespace(status=200)))
    monkeypatch.setattr(sweep.subprocess, "check_output", lambda *a, **k: " M mstar/worker/worker.py\n")

    def forbidden(*args, **kwargs):
        pytest.fail("uncommitted code was benchmarked")

    monkeypatch.setattr(sweep.subprocess, "run", forbidden)
    with pytest.raises(RuntimeError, match="uncommitted"):
        sweep.main(["--out", str(tmp_path / "later")])


@pytest.mark.parametrize('suite, count', [('qwen', 4), ('streaming', 4), ('tp-async', 4), ('tcp', 2), ('all', 18)])
def test_a40_schedule_dry_run_is_read_only(tmp_path, monkeypatch, capsys, suite, count):
    from benchmark.transport.a40 import run

    def forbidden(*args, **kwargs):
        pytest.fail('dry-run executed a subprocess')

    monkeypatch.setattr(run.subprocess, 'check_output', forbidden)
    monkeypatch.setattr(run.subprocess, 'run', forbidden)
    out = tmp_path / 'results'
    assert run.main(['--root', str(tmp_path / 'root'), '--out', str(out), '--suite', suite, '--dry-run']) == 0
    output = capsys.readouterr().out
    assert output.count('--tensor-comm-protocol') == count
    assert not out.exists()


def test_a40_commands_use_each_arms_config_and_separate_tcp_smoke(tmp_path):
    from benchmark.transport.a40.run import commands

    base, fix, case = [tmp_path / name for name in ('base', 'fix', 'case')]
    server, client = commands(base, fix, case, 'qwen', 8100, [1, 32, 64])
    assert str(base / 'configs/qwen3_5_0.8b.yaml') in server
    assert str(fix / 'benchmark/transport/request_cleanup_gpu.py') in client
    assert client[client.index('--checkout') + 1] == str(base)
    server, client = commands(base, fix, case, 'tcp', 8100, [1, 32, 64])
    assert server[-1] == 'TCP'
    assert client[0] == 'bash'


def test_a40_startup_failure_still_stops_owned_server(tmp_path, monkeypatch):
    from benchmark.transport.a40 import run

    calls = []
    monkeypatch.setattr(run, 'capture', lambda *args: 'source')

    def execute(command, **kwargs):
        calls.append(command)
        if 'start' in command:
            raise RuntimeError('startup failed')

    monkeypatch.setattr(run.subprocess, 'run', execute)
    with pytest.raises(RuntimeError, match='startup failed'):
        run.run_case(tmp_path, tmp_path / 'base', tmp_path / 'fix', tmp_path / 'case',
                     'qwen', 8100, [1], 'rust')
    assert [command[2] for command in calls] == ['start', 'stop']


def test_a40_comparison_preserves_repeated_measurements(tmp_path):
    from benchmark.transport.a40.run import summarize

    for index, arm in enumerate(('base', 'fix', 'fix', 'base'), 1):
        case = tmp_path / 'qwen' / f'{index}-{arm}'
        (case / 'clients').mkdir(parents=True)
        (case / 'metadata.json').write_text(json.dumps({'scenario': 'qwen'}))
        (case / 'clients/summary.json').write_text(json.dumps([
            {'case': 'c64-tokens16', 'request_throughput': 10.0 + index,
             'jct_median_ms': 20.0, 'aggregate': {'total_text_tokens': 1024}},
        ]))
    summarize(tmp_path)
    rows = json.loads((tmp_path / 'comparison.json').read_text())
    assert len(rows) == 1
    assert rows[0]['request_throughput'] == {'base': [11.0, 14.0], 'fix': [12.0, 13.0]}
    assert rows[0]['request_throughput_median_change_pct'] == 0.0
    assert len(rows[0]['repeats']['fix']) == 2


@pytest.mark.parametrize('fail_client', [False, True])
def test_a40_tcp_client_and_monitor_cleanup(tmp_path, monkeypatch, fail_client):
    from benchmark.transport.a40 import run

    calls = []
    monkeypatch.setattr(run, 'capture', lambda *args: 'source')

    class Monitor:
        def terminate(self):
            calls.append('terminate-monitor')

        def wait(self, timeout):
            calls.append('reap-monitor')

    monkeypatch.setattr(run.subprocess, 'Popen', lambda *a, **k: Monitor())

    def execute(command, **kwargs):
        if 'start' in command or 'stop' in command:
            calls.append(command[2])
            return
        calls.append('client')
        if fail_client:
            raise RuntimeError('client failed')
        output = tmp_path / 'case/clients'
        output.mkdir()
        (output / 'results.json').write_text(json.dumps({
            'completed': 2, 'failed': 0, 'request_throughput': 2.0,
            'aggregate': {'max_concurrency': 1},
        }))
        assert kwargs['env']['N'] == '2'
        assert kwargs['env']['CONC'] == '1'
        assert kwargs['env']['MSTAR_TP_ASYNC_SCHED'] == '0'

    monkeypatch.setattr(run.subprocess, 'run', execute)
    context = pytest.raises(RuntimeError, match='client failed') if fail_client else nullcontext()
    with context:
        run.run_case(tmp_path, tmp_path / 'base', tmp_path / 'fix', tmp_path / 'case',
                     'tcp', 8100, [1, 32, 64], 'rust')
    assert calls == ['start', 'client', 'terminate-monitor', 'reap-monitor', 'stop']
    assert (tmp_path / 'case/PASS').exists() is not fail_client
