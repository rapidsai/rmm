# REPLAY_BENCH

`REPLAY_BENCH` reads an RMM memory event log, a CSV file written by
`rmm::mr::logging_resource_adaptor`, and replays every allocation and free
against the chosen memory resource. Use it to reproduce an allocator
failure without the original workload or the original GPU.

The tool is a benchmark. It is not in any conda package or pip wheel. Build RMM
from the source root with `./build.sh librmm benchmarks`.

The binary is `cpp/build/gbenchmarks/REPLAY_BENCH`.

```bash
./cpp/build/gbenchmarks/REPLAY_BENCH -f rmm-log.txt -r pool -s 78 --benchmark_min_time=1x
```

Always pass `-r`, use a whole number for `-s`, and read the output text. The
exit code is always 0.

For the full guide, including how to record a log, see
[Record and Replay Memory Events](../../../docs/user_guide/record_replay.md).
