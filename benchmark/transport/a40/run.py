"""Serial, restart-controlled two-A40 benchmarks using the existing repo clients."""

import argparse
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
from pathlib import Path

BASE = '9f84f18cd7e1b4f3851d0bde85b8da45cabec06f'
# name, model, config, GPUs, TP async, protocol
SCENARIOS = {
    'qwen': ('qwen3_5_0.8b', 'qwen3_5_0.8b.yaml', '0', '0', 'SHM'),
    'streaming': ('orpheus', 'orpheus_pd_disaggregated.yaml', '0,1', '0', 'SHM'),
    'tp-sync': ('orpheus', 'orpheus_tp2.yaml', '0,1', '0', 'SHM'),
    'tp-async': ('orpheus', 'orpheus_tp2.yaml', '0,1', '1', 'SHM'),
    'tcp': ('orpheus', 'orpheus_pd_disaggregated.yaml', '0,1', '0', 'TCP'),
}


def capture(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def environment(root, checkout, runtime, asynchronous):
    return dict(os.environ, PYTHONPATH=str(checkout),
                PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''),
                HF_HOME=str(root / 'cache/huggingface'),
                TORCHINDUCTOR_CACHE_DIR=str(root / 'cache/inductor'),
                TRITON_CACHE_DIR=str(root / 'cache/triton'),
                MSTAR_RUST_GRAPH=str(int(runtime == 'rust')),
                MSTAR_RUST_ZMQ=str(int(runtime == 'rust')),
                MSTAR_SHM_ARENA='1', MSTAR_TP_ASYNC_SCHED=asynchronous,
                MSTAR_PHASE_TIMING='10', OMP_NUM_THREADS='2')


def commands(checkout, fix, case, scenario, port, concurrency):
    model, config, gpus, _, protocol = SCENARIOS[scenario]
    server = [sys.executable, '-m', 'mstar.cli.main', 'serve', model,
              '--config', str(checkout / 'configs' / config), '--gpus', gpus,
              '--port', str(port), '-O2', '--tensor-comm-protocol', protocol]
    if scenario == 'tcp':
        client = ['bash', str(checkout / 'benchmark/worker_phases/clients/orpheus_t2s.sh'),
                  '--output-dir', str(case / 'clients')]
    else:
        client = [sys.executable, str(fix / 'benchmark/transport/request_cleanup_gpu.py'),
                  '--checkout', str(checkout), '--out', str(case / 'clients'),
                  '--url', f'http://127.0.0.1:{port}',
                  '--model', 'qwen' if scenario == 'qwen' else 'orpheus',
                  '--concurrency', *map(str, concurrency)]
    return server, client


def run_case(root, checkout, fix, case, scenario, port, concurrency, runtime):
    from benchmark.transport.request_cleanup_gpu import read_result
    from benchmark.worker_phases.phases import parse_log, render, segment

    case.mkdir()
    env = environment(root, checkout, runtime, SCENARIOS[scenario][3])
    # A unique state directory prevents interference with another benchmark session.
    env['SERVER_STATE_DIR'] = str(case / 'server-state')
    server, client = commands(checkout, fix, case, scenario, port, concurrency)
    lifecycle = ['bash', str(fix / '.claude/skills/benchmarking/scripts/server.sh')]
    metadata = {
        'checkout': str(checkout), 'commit': capture(['git', 'rev-parse', 'HEAD'], checkout),
        'scenario': scenario, 'runtime': runtime, 'server': server, 'client': client,
        'environment': {k: env[k] for k in env if k.startswith(('MSTAR_', 'OMP_', 'HF_', 'TORCHINDUCTOR_', 'TRITON_'))},
        'source': capture([sys.executable, '-c', 'import mstar; print(mstar.__file__)'], checkout),
    }
    (case / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    monitor = None
    with (case / 'lifecycle.log').open('w') as lifecycle_log:
        try:
            subprocess.run([*lifecycle, 'start', 'bench', str(port), '--timeout', '3600',
                            '--log', str(case / 'server.log'), '--', *server],
                           cwd=checkout, env=env, stdout=lifecycle_log, stderr=subprocess.STDOUT, check=True)
            with (case / 'gpu.csv').open('w') as gpu_log:
                monitor = subprocess.Popen([
                    'nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw,clocks.sm',
                    '--format=csv', '-l', '5'], stdout=gpu_log, stderr=subprocess.STDOUT)
                if scenario == 'tcp':
                    env.update(URL=f'http://127.0.0.1:{port}', CONC='1', N='2', WARMUP='1')
                with (case / 'sweep.log').open('w') as client_log:
                    subprocess.run(client, cwd=checkout, env=env, stdout=client_log,
                                   stderr=subprocess.STDOUT, check=True)
                if scenario == 'tcp':
                    read_result(case / 'clients/results.json', expected=2, concurrency=1)
        finally:
            try:
                if monitor is not None:
                    monitor.terminate()
                    try:
                        monitor.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        monitor.kill()
                        monitor.wait(timeout=10)
            finally:
                subprocess.run([*lifecycle, 'stop', 'bench'], cwd=checkout, env=env,
                               stdout=lifecycle_log, stderr=subprocess.STDOUT, check=True)
            log = case / 'server.log'
            if log.exists():
                (case / 'phases.txt').write_text(render(segment(parse_log(log.read_text()))) + '\n')
    (case / 'PASS').write_text('Clients completed; inspect logs and compare repeats before acceptance.\n')


def summarize(out):
    grouped = {}
    for manifest in sorted(out.glob('*/*/metadata.json')):
        meta = json.loads(manifest.read_text())
        summary = manifest.parent / 'clients/summary.json'
        if not summary.exists():  # TCP is a functional smoke, not a timing result.
            continue
        arm = manifest.parent.name.split('-')[1]
        for cell in json.loads(summary.read_text()):
            key = (meta['scenario'], cell['case'])
            grouped.setdefault(key, {}).setdefault(arm, []).append(cell)
    rows = []
    for (scenario, case), arms in sorted(grouped.items()):
        row = {'scenario': scenario, 'case': case, 'repeats': arms}
        for metric in ('request_throughput', 'jct_median_ms'):
            values = {arm: [c[metric] for c in cells] for arm, cells in arms.items()}
            row[metric] = values
            if set(values) == {'base', 'fix'}:
                row[metric + '_median_change_pct'] = 100 * (
                    statistics.median(values['fix']) / statistics.median(values['base']) - 1)
        rows.append(row)
    (out / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='setup.sh output directory')
    parser.add_argument('--out', type=Path, required=True, help='new results directory')
    parser.add_argument('--suite', nargs='+', choices=['all', *SCENARIOS], default=['all'])
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 32, 64])
    parser.add_argument('--runtime', choices=['rust', 'python'], default='rust')
    parser.add_argument('--port', type=int, default=8100)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if min(args.concurrency) < 1:
        parser.error('concurrency must be positive')
    root, out = args.root.resolve(), args.out.resolve()
    suites = list(SCENARIOS) if 'all' in args.suite else list(dict.fromkeys(args.suite))
    # A dry-run only describes commands; no GPU, server, git or filesystem mutation.
    if not args.dry_run:
        import torch

        if torch.cuda.device_count() != 2 or any('A40' not in torch.cuda.get_device_name(i) for i in range(2)):
            raise RuntimeError('Expected exactly two visible A40 GPUs')
        if os.statvfs('/dev/shm').f_bavail * os.statvfs('/dev/shm').f_frsize < 8 * 1024**3:
            raise RuntimeError('Need at least 8 GiB available in /dev/shm (16 GiB recommended)')
        if capture(['git', 'rev-parse', 'HEAD'], root / 'base') != BASE:
            raise RuntimeError('Base checkout is not the pinned baseline')
        for arm in ('base', 'fix'):
            if capture(['git', 'status', '--porcelain', '--untracked-files=no'], root / arm):
                raise RuntimeError(f'{arm} checkout has uncommitted tracked changes')
        if capture(['git', 'rev-parse', 'HEAD'], root / 'fix') == BASE:
            raise RuntimeError('Fix and base identify the same commit')
        out.mkdir(parents=True, exist_ok=False)
        (out / 'hardware.txt').write_text(capture(['nvidia-smi', '-q']) + '\n' + capture(['nvidia-smi', 'topo', '-m']))
    for scenario in suites:
        sequence = ['base', 'fix'] if scenario == 'tcp' else ['base', 'fix', 'fix', 'base']
        if not args.dry_run:
            (out / scenario).mkdir()
        for index, arm in enumerate(sequence, 1):
            case = out / scenario / f'{index}-{arm}'
            server, client = commands(root / arm, root / 'fix', case, scenario, args.port, args.concurrency)
            print(f'{scenario} {index}-{arm}: {shlex.join(server)}\n  {shlex.join(client)}', flush=True)
            if not args.dry_run:
                run_case(root, root / arm, root / 'fix', case, scenario, args.port, args.concurrency, args.runtime)
    if not args.dry_run:
        summarize(out)
    return 0


def interrupt(signum, frame):
    raise KeyboardInterrupt


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    raise SystemExit(main())
