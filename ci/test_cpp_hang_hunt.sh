#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# DO NOT MERGE: loop the C++ test suite and gdb-dump any test process that hangs.

set -euo pipefail

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

conda config --set channel_priority strict

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp librmm rmm --cuda "$RAPIDS_CUDA_VERSION")")

rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test
rapids-mamba-retry install --yes -n test -c conda-forge gdb gcc

set +u
conda activate test
set -u

rapids-print-env
./ci/gpu_info.sh
cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo "ptrace_scope: not present"

ITERS=${HANG_HUNT_ITERS:-12}
PARALLEL=${HANG_HUNT_PARALLEL:-4}
BURST=${HANG_HUNT_BURST:-8}
BURST_ROUNDS=${HANG_HUNT_BURST_ROUNDS:-20}
HOG_PERCENT=${HANG_HUNT_HOG_PERCENT:-60}
HANG_S=${HANG_HUNT_HANG_S:-180}
DEADLINE=$((SECONDS + ${HANG_HUNT_MINUTES:-100} * 60))
GTESTS="${CONDA_PREFIX}/bin/gtests/librmm"
LOGS=$(mktemp -d)
hangs=0
iter=0

# Dump every test process older than HANG_S, then kill it so the suite can finish.
watch_tests() {
  local p age
  # ctest launches each test as a relative path, e.g. ./../../..//bin/gtests/librmm/ARENA_MR_REF_TEST.
  for p in $(pgrep -f "^[^ ]*/gtests/librmm/[A-Z_]*_TEST( |$)" || true); do
    age=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ') || continue
    [[ -n "$age" && "$age" -gt "$HANG_S" ]] || continue
    echo "::group::HANG iteration ${iter}: pid ${p} $(tr '\0' ' ' < "/proc/${p}/cmdline") (${age} s)"
    grep -E '^(State|Threads|voluntary_ctxt_switches|nonvoluntary_ctxt_switches)' "/proc/${p}/status" || true
    for t in /proc/"${p}"/task/*; do
      echo "task ${t##*/}: $(cat "${t}/comm") state=$(awk '{print $3}' "${t}/stat") wchan=$(cat "${t}/wchan" 2>/dev/null)"
    done
    cat "/proc/${p}/stack" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv || true
    timeout 120 gdb -p "$p" -batch -ex 'info threads' -ex 'thread apply all bt' 2>&1 || echo "gdb failed: $?"
    echo "::endgroup::"
    kill -9 "$p" || true
    hangs=$((hangs + 1))
  done
}

# The CI container has yama ptrace_scope=1 and no CAP_SYS_PTRACE, so gdb can only attach to tests that allow it.
# A test that crashes also gets a gdb backtrace of all threads before it dies.
cat > "${LOGS}/ptracer.c" <<'EOF'
#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>
static char self[24];
static void on_crash(int sig) {
  pid_t child = fork();
  if (child == 0) {
    execlp("gdb", "gdb", "-p", self, "-batch", "-ex", "info threads", "-ex", "thread apply all bt", (char*)0);
    _exit(127);
  }
  if (child > 0) waitpid(child, 0, 0);
  signal(sig, SIG_DFL);
  raise(sig);
}
__attribute__((constructor)) static void setup(void) {
  prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
  int n = 0;
  for (pid_t p = getpid(); p > 0; p /= 10) n++;
  for (pid_t p = getpid(); n > 0; p /= 10) self[--n] = '0' + p % 10;
  signal(SIGSEGV, on_crash);
  signal(SIGBUS, on_crash);
  signal(SIGABRT, on_crash);
}
EOF
gcc -shared -fPIC -o "${LOGS}/libptracer.so" "${LOGS}/ptracer.c"

cd "${GTESTS}"
export LD_PRELOAD="${LOGS}/libptracer.so"

# Wait for the test processes in `pids` while dumping any test that hangs.
pids=()
wait_watching() {
  local p
  for p in "${pids[@]}"; do
    while kill -0 "${p}" 2>/dev/null; do
      sleep 5
      watch_tests
    done
    wait "${p}" || true
  done
  pids=()
}

# Phase 1: right at the runner's first GPU use, when both nightly hangs hit, start bursts of BURST simultaneous
# launches of each small test that hung in the nightlies.
round=0
while (( round < BURST_ROUNDS && SECONDS < DEADLINE )); do
  round=$((round + 1))
  iter="burst${round}"
  start=$SECONDS
  for t in SYSTEM_MR_REF_PTDS_TEST DEVICE_SCALAR_PTDS_TEST BINNING_MR_PTDS_TEST PINNED_POOL_MR_REF_TEST \
           PINNED_POOL_MR_REF_PTDS_TEST DEVICE_BUFFER_TEST; do
    for c in $(seq 1 "${BURST}"); do
      "./${t}" > "${LOGS}/${iter}_${t}_${c}.log" 2>&1 &
      pids+=($!)
    done
  done
  wait_watching
  failed=$(grep -L 'PASSED' "${LOGS}/${iter}"_*.log | wc -l)
  echo "burst ${round}: $((SECONDS - start)) s, ${failed} processes without PASSED"
  for f in $(grep -L 'PASSED' "${LOGS}/${iter}"_*.log | head -3); do
    echo "::group::${f}"
    tail -30 "${f}"
    echo "::endgroup::"
  done
done

# GB300 memory is coherent with the host (ATS), so neighbouring jobs' host memory pressure reaches the GPU driver.
cat > "${LOGS}/hog.c" <<'EOF'
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char** argv) {
  size_t n = strtoull(argv[1], 0, 10) << 20, step = 64ull << 20;
  char* p = malloc(n);
  for (;;) for (size_t i = 0; i < n; i += step) { memset(p + i, i, step < n - i ? step : n - i); usleep(1000); }
}
EOF
gcc -O1 -o "${LOGS}/hog" "${LOGS}/hog.c"
avail_mb=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
# The pod's cgroup limit can be far below the node's MemAvailable; stay inside it.
if [[ -r /sys/fs/cgroup/memory.max && "$(cat /sys/fs/cgroup/memory.max)" != max ]]; then
  cg_mb=$(( ($(cat /sys/fs/cgroup/memory.max) - $(cat /sys/fs/cgroup/memory.current)) / 1048576 ))
  (( cg_mb < avail_mb )) && avail_mb=${cg_mb}
fi
echo "MemAvailable: $(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo) MiB, usable ${avail_mb} MiB;" \
  "hog takes $((avail_mb * HOG_PERCENT / 100)) MiB"
LD_PRELOAD='' "${LOGS}/hog" $((avail_mb * HOG_PERCENT / 100)) &
hog=$!

iter=0
# Phase 2: under host memory pressure, PARALLEL concurrent copies of the whole suite, so many test processes create CUDA contexts at once.
while (( iter < ITERS && SECONDS < DEADLINE )); do
  iter=$((iter + 1))
  start=$SECONDS
  for c in $(seq 1 "${PARALLEL}"); do
    ctest --no-tests=error --output-on-failure -j20 > "${LOGS}/iter_${iter}_${c}.log" 2>&1 &
    pids+=($!)
  done
  wait_watching
  for c in $(seq 1 "${PARALLEL}"); do
    log="${LOGS}/iter_${iter}_${c}.log"
    echo "iteration ${iter}.${c}: $((SECONDS - start)) s $(grep -o '[0-9]*% tests passed.*' "${log}" || echo 'no summary')"
    if grep -q '\*\*\*' "${log}"; then
      echo "::group::ctest log for iteration ${iter}.${c}"
      cat "${log}"
      echo "::endgroup::"
    fi
  done
done

kill "${hog}" || true

rapids-logger "Hang hunt: ${hangs} hung test processes"
[[ "${hangs}" -eq 0 ]]
