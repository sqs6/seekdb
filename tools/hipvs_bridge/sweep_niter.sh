#!/usr/bin/env bash
# niter 扫描：自动找出「Recall 降幅达标的前提下最快」的那一档，并给出推荐值。
#
# 为什么需要它：`nn_descent_niter` 是本题里唯一同时决定「建图速度」和「Recall」的旋钮，
# 而 Hidden250K（250K×256，权重 60%）内容不公开、基线 Recall 更低（0.985 / 0.914），
# 降幅只能在真机上量。官方硬门槛是「Recall 相对同轮基线下降 ≤ 0.005」，超了就是本次总分 0。
#
# 用法（在带 AMD GPU + ROCm + hipVS 的机器上）：
#   bash tools/hipvs_bridge/sweep_niter.sh --rows 250000 --dim 256 --nq 4096
#
# 可选环境变量：
#   NITERS="10 15 20 30 50"      要扫描的 niter 列表
#   OBSERVER_START=...            启动 observer 的命令（默认用云端模板的 start_observer.sh）
#   OBSERVER_STOP=...             停 observer 的命令（默认 pkill -f observer）
set -uo pipefail

TOPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR=${OUT_DIR:-/tmp/amd_sweep}
NITERS=${NITERS:-"10 15 20 30 50"}
ROWS=250000
DIM=256
NQ=4096
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --rows) ROWS="$2"; shift 2;;
    --dim)  DIM="$2";  shift 2;;
    --nq)   NQ="$2";   shift 2;;
    --extra) EXTRA+=("$2"); shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

mkdir -p "$OUT_DIR"
cd "$TOPDIR" || exit 1

# ---- observer 的启停 ------------------------------------------------------
stop_observer() {
  if [ -n "${OBSERVER_STOP:-}" ]; then
    eval "$OBSERVER_STOP" >/dev/null 2>&1
  else
    pkill -f 'observer' >/dev/null 2>&1
  fi
  sleep 3
}

start_observer() {
  stop_observer
  local cmd="${OBSERVER_START:-}"
  if [ -z "$cmd" ]; then
    for c in "${SEEKDB_HOME:-/seekdb_workspace}/scripts/start_observer.sh" \
             "tools/deploy/start_observer.sh"; do
      if [ -x "$c" ]; then cmd="bash $c $OUT_DIR/run"; break; fi
    done
  fi
  if [ -z "$cmd" ]; then
    echo "找不到启动 observer 的方式，请设置 OBSERVER_START" >&2
    return 1
  fi
  echo "  启动 observer: $cmd"
  eval "$cmd" >/dev/null 2>&1
  for _ in $(seq 1 120); do
    if mysql -h127.0.0.1 -P2881 -uroot -e "select 1" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "observer 没起来" >&2
  return 1
}

run_one() {   # $1=preset $2=niter(可空) $3=输出 json
  local preset="$1" niter="$2" out="$3"
  if [ -n "$niter" ]; then export SEEKDB_CUVS_NITER="$niter"; else unset SEEKDB_CUVS_NITER; fi
  export SEEKDB_CUVS_PRESET="$preset"
  echo "--- 轮次：preset=$preset niter=${niter:-默认} ---"
  start_observer || return 1
  python3 tools/hipvs_bridge/bench_cagra.py \
      --rows "$ROWS" --dim "$DIM" --nq "$NQ" --preset "$preset" \
      --out "$out" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tail -20
}

echo "===== [1/3] 基线轮（冻结基线行为）====="
run_one baseline "" "$OUT_DIR/baseline.json" || { echo "基线轮失败"; exit 1; }

echo "===== [2/3] 扫描 niter：$NITERS ====="
RUNS=""
for n in $NITERS; do
  run_one "niter-$n" "$n" "$OUT_DIR/niter_$n.json" || { echo "niter=$n 这一轮失败，跳过"; continue; }
  RUNS="$RUNS,$OUT_DIR/niter_$n.json"
done
RUNS=${RUNS#,}

echo "===== [3/3] 汇总与推荐 ====="
python3 tools/hipvs_bridge/bench_cagra.py --summary \
    --baseline "$OUT_DIR/baseline.json" --runs "$RUNS" --margin "${MARGIN:-0.003}"
echo
echo "结果目录：$OUT_DIR"
echo "把推荐值填进 tools/hipvs_bridge/seekdb_cuvs_bridge.c 的 env_long(\"SEEKDB_CUVS_NITER\", <推荐值>, 1, 1000) 默认值，重编译后提交。"
echo "★ 两个维度都要跑一遍（--dim 128 与 --dim 256），两档都达标才能提交；"
echo "  Hidden250K 占 60% 权重且更敏感，以它的结论为准。"
