"""Bounded A40 follow-up using the existing lifecycle, clients and result checks.

This checks streaming at c64 in ABBA order and TP sync/async on both versions.
The reduced workload is a targeted follow-up, not a full performance sweep.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

# The control checkout supplies tooling; child processes import their own arm.
CONTROL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CONTROL))

from benchmark.transport.a40 import run  # noqa: E402
from benchmark.transport.request_cleanup_gpu import read_result  # noqa: E402


def client(checkout, case, scenario, port):
    cells = [(64, 128, 2)] if scenario == 'streaming' else [(1, 2, 1), (64, 64, 1)]
    summaries = []
    for concurrency, requests, warmup in cells:
        out = case / f'c{concurrency}'
        env = dict(os.environ, URL=f'http://127.0.0.1:{port}', CONC=str(concurrency),
                   N=str(requests), WARMUP=str(warmup))
        command = ['bash', str(checkout / 'benchmark/worker_phases/clients/orpheus_t2s.sh'),
                   '--output-dir', str(out)]
        out.mkdir(parents=True)
        with (out / 'client.log').open('w') as log:
            subprocess.run(command, cwd=checkout, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        result = read_result(out / 'results.json', requests, concurrency)
        summaries.append(dict(case=out.name, completed=result['completed'],
                              request_throughput=result['request_throughput'],
                              jct_median_ms=result['jct_median_ms'], aggregate=result['aggregate']))
        (case / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8100)
    parser.add_argument('--budget-seconds', type=int, default=1800)
    parser.add_argument('--suite', nargs='+', choices=['streaming', 'tp-sync', 'tp-async'],
                        default=['streaming', 'tp-sync', 'tp-async'])
    parser.add_argument('--client', choices=['streaming', 'tp-sync', 'tp-async'])
    parser.add_argument('--checkout', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.client:
        client(args.checkout, args.out, args.client, args.port)
        return
    jobs = [(scenario, index, arm)
            for scenario in dict.fromkeys(args.suite)
            for index, arm in enumerate(('base', 'fix', 'fix', 'base')
                                       if scenario == 'streaming' else ('base', 'fix'), 1)]
    for scenario, index, arm in jobs:
        print(scenario, index, arm, flush=True)
    if args.dry_run:
        return
    args.out.mkdir(parents=True)
    deadline = time.monotonic() + args.budget_seconds
    original_commands, original_environment = run.commands, run.environment

    def commands(checkout, fix, case, scenario, port, concurrency):
        server, _ = original_commands(checkout, fix, case, scenario, port, concurrency)
        driver = [sys.executable, str(Path(__file__).resolve()), '--root', str(args.root),
                  '--out', str(case / 'clients'), '--checkout', str(checkout),
                  '--client', scenario, '--port', str(port)]
        print('SERVER', shlex.join(server), 'CLIENT', shlex.join(driver), flush=True)
        return server, driver

    def environment(root, checkout, runtime, asynchronous):
        env = original_environment(root, checkout, runtime, asynchronous)
        # Fresh Inductor/Triton caches per boot, with the downloaded weights warm.
        env.update(TORCHINDUCTOR_CACHE_DIR=str(case / 'cache/inductor'),
                   TRITON_CACHE_DIR=str(case / 'cache/triton'))
        if scenario.startswith('tp-'):
            env['NCCL_P2P_DISABLE'] = '1'
            env['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '60'
        return env

    run.commands, run.environment = commands, environment
    checkpoint = []
    for scenario, index, arm in jobs:
        case = args.out / scenario / f'{index}-{arm}'
        case.parent.mkdir(exist_ok=True)
        checkout = args.root / f'current-{arm}'
        if run.capture(['git', 'status', '--porcelain', '--untracked-files=no'], checkout):
            raise RuntimeError(f'{arm} checkout is dirty')
        entry = dict(scenario=scenario, arm=arm, case=str(case), status='running')
        remaining = int(deadline - time.monotonic())
        if remaining < 60:
            entry.update(status='not_started', error='Rental time budget exhausted')
        checkpoint.append(entry)
        (args.out / 'checkpoint.json').write_text(json.dumps(checkpoint, indent=2) + '\n')
        if remaining < 60:
            raise TimeoutError('Rental time budget exhausted')
        signal.alarm(min(420, remaining))
        try:
            run.run_case(args.root, checkout, args.root / 'current-fix', case,
                         scenario, args.port, [64], 'rust')
            entry['status'] = 'passed'
        except BaseException as exc:
            entry.update(status='failed', error=repr(exc))
            raise
        finally:
            signal.alarm(0)
            (args.out / 'checkpoint.json').write_text(json.dumps(checkpoint, indent=2) + '\n')
    run.summarize(args.out)


if __name__ == '__main__':
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGALRM):
        signal.signal(sig, run.interrupt)
    main()
