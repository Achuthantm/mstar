# Failure excerpts and diagnostic outcomes

Repeated stack traces and repeated rank messages are omitted. Original source paths and line numbers identify the evidence.

## Expected baseline race reproduction

Source: `base-race.log`

```text
L297: >               pytest.fail(f"Race reproduced: cleanup invalidated {engine.invalid_endpoints} before read finished: {exc}")
L2408: 8 failed, 4 passed in 0.89s
```

## Streaming baseline c64: one disconnected request

Source: `results/reduced-streaming-1/streaming/4-base/clients/c64/client.log`

```text
L7: Requests : 319/320 succeeded
L19: --- Errors (1/320) ---
L20:   [0] Server disconnected
```

## TP sync baseline startup

Source: `results/reduced-tp-sync-1/tp-sync/1-base/server.log`

```text
L70: [rank1]:[E1101 08:13:10.125933177 ProcessGroupNCCL.cpp:688] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600088 milliseconds before timing out.
L142: 2026-11-01 08:14:10,879 ERROR [conductor] mstar.conductor.conductor: Worker worker_1 (pid 22049) exited with signal SIGABRT. It hosted nodes ['LLM']. Shutting the deployment down.
```

## TP async baseline startup

Source: `results/reduced-tp-async-1/tp-async/1-base/server.log`

```text
L72: [rank1]:[E1101 08:24:44.309540819 ProcessGroupNCCL.cpp:688] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600098 milliseconds before timing out.
L114: 2026-11-01 08:25:44,618 ERROR [conductor] mstar.conductor.conductor: Worker worker_1 (pid 23532) exited with signal SIGABRT. It hosted nodes ['LLM']. Shutting the deployment down.
```

## Initial Rust TCP dependency failure

Source: `results/reduced-tcp-1/tcp/1-base/server.log`

```text
L45: RuntimeError: Mooncake TransferEngine is required for RDMA/TCP protocol. Failed to load mooncake: ImportError: libibverbs.so.1: cannot open shared object file: No such file or directory. Install mooncake-transfer-engine or use SHM protocol.
```

## Standalone default NCCL reproduction

Source: `diagnostics/nccl-default.log`

```text
L73: [rank1]:[E1101 08:34:40.626331856 ProcessGroupNCCL.cpp:688] [Rank 1] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=1, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=30000) ran for 30025 milliseconds before timing out.
```

## Standalone NCCL with P2P disabled: successful diagnostic

Source: `diagnostics/nccl-no-p2p.log`

```text
L69: PASS rank=0 all_reduce=3.0
L70: PASS rank=1 all_reduce=3.0
```

## Rust TCP retry outcome

After installing `libibverbs1` and `librdmacm1`, both final Rust TCP cases passed 2/2 measured requests. See `results/final-validation/tcp-1-base-attempt1` and `tcp-2-fix-attempt1`.

## Cancellation

The final streaming rerun and all subsequent TP model retries were cancelled by the user after Rust TCP completed. These are not passes. The standalone P2P-disabled NCCL pass is not a model TP validation.

