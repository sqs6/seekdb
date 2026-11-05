#!/usr/bin/env bash
# AMD 赛题本地复现脚本：编译 -> 起 observer -> 跑 baseline / candidate 两轮 -> 对比。
#
# 用法（在带 AMD GPU + ROCm + hipVS 的机器上）：
#   cd <答题仓库根目录>
#   bash tools/hipvs_bridge/bench_cagra.sh --rows 250000 --dim 128 --nq 4096
#
# 说明：
#   * 编译命令与官方评测完全一致（见《AMD 赛题说明》三、统一编译方式）；
#   * baseline 轮通过环境变量 SEEKDB_CUVS_PRESET=baseline 让 bridge 精确复现冻结基线
#     （niter=100、itopk 下限 512、每次查询独立分配显存），candidate 轮是优化后的默认行为；
#   * 两轮在同一台机器、同一份数据上顺序执行，算出的加速比与 recall 差值可直接对照官方口径。

set -uo pipefail

TOPDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROCM_ROOT=${ROCM_ROOT:-/opt/rocm}
HIPVS_ROOT=${HIPVS_ROOT:-/opt/hipvs}
OUT_DIR=${OUT_DIR:-/tmp/amdbench}
ROWS=250000
DIM=128
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

echo "===== [1/5] 编译 hipVS bridge ====="
HIPVS_ROOT="$HIPVS_ROOT" ROCM_ROOT="$ROCM_ROOT" \
  tools/hipvs_bridge/build.sh "$OUT_DIR/libseekdb_cuvs_bridge.so" || exit 1

echo "===== [2/5] 编译 seekdb（官方参数）====="
cmake -S . -B build_release -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DOB_ENABLE_UNITY=ON \
  -DOB_USE_LLD=ON \
  -DOB_BUILD_CUVS=ON \
  -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB="$OUT_DIR/libseekdb_cuvs_bridge.so" || exit 1
cmake --build build_release --target seekdb -j "$(nproc)" || exit 1

echo "===== [3/5] 起 observer ====="
BASE_RUN="$OUT_DIR/run"
rm -rf "$BASE_RUN"; mkdir -p "$BASE_RUN"
if [ -x "${SEEKDB_HOME:-/seekdb_workspace}/scripts/start_observer.sh" ]; then
  "${SEEKDB_HOME}/scripts/start_observer.sh" "$BASE_RUN" || exit 1
elif [ -x tools/deploy/start_observer.sh ]; then
  tools/deploy/start_observer.sh "$BASE_RUN" || exit 1
else
  echo "找不到 start_observer.sh，请先手动起 seekdb observer（127.0.0.1:2881）。"
fi
for _ in $(seq 1 180); do
  if mysql -h127.0.0.1 -P2881 -uroot -e "select 1" >/dev/null 2>&1; then break; fi
  sleep 1
done

echo "===== [4/5] baseline 轮（冻结基线行为）====="
SEEKDB_CUVS_PRESET=baseline SEEKDB_CUVS_TRACE=1 \
  python3 tools/hipvs_bridge/bench_cagra.py \
    --rows "$ROWS" --dim "$DIM" --nq "$NQ" --preset baseline \
    --load-file "$OUT_DIR/load.csv" --out "$OUT_DIR/baseline.json" "${EXTRA[@]+"${EXTRA[@]}"}" \
  2>&1 | tee "$OUT_DIR/baseline.log" || exit 1

echo "===== [5/5] candidate 轮（优化后）====="
SEEKDB_CUVS_PRESET=candidate SEEKDB_CUVS_TRACE=1 \
  python3 tools/hipvs_bridge/bench_cagra.py \
    --rows "$ROWS" --dim "$DIM" --nq "$NQ" --preset candidate \
    --load-file "$OUT_DIR/load.csv" --out "$OUT_DIR/candidate.json" "${EXTRA[@]+"${EXTRA[@]}"}" \
  2>&1 | tee "$OUT_DIR/candidate.log" || exit 1

echo "===== 对比 ====="
python3 tools/hipvs_bridge/bench_cagra.py --compare "$OUT_DIR/baseline.json" "$OUT_DIR/candidate.json"
echo "结果目录：$OUT_DIR"
