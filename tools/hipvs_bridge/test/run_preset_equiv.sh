#!/usr/bin/env bash
# 验证「本实现的 baseline 预设」与「冻结基线的 bridge」**逐项一致**。
#
# 为什么重要：整套 A/B 方法论（拿 baseline 预设当对照组、用它算加速比与 Recall 差值）
# 都建立在这个等式上。若两者参数不一致，本地量出来的加速比就是假的，
# 由此推荐的 niter 也会是错的。
#
# 做法：把冻结基线那份 bridge（git show 634888d9:tools/hipvs_bridge/seekdb_cuvs_bridge.c）
# 与本实现的 bridge（带 SEEKDB_CUVS_PRESET=baseline）各自用同一套 cuVS 桩实现编成可执行文件，
# 跑同一个 driver，比较两边桩观测到的 建图参数 / 检索参数 / 结果形状，必须完全一致。
#
# 用法（在仓库根目录）：bash tools/hipvs_bridge/test/run_preset_equiv.sh
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BRIDGE_DIR=$(cd "$HERE/.." && pwd)
REPO=$(cd "$BRIDGE_DIR/../.." && pwd)
V5=634888d9b037153fa143d2c913ba4b25503e1d62
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "=== 1. 取出冻结基线的 bridge ==="
(cd "$REPO" && git show "$V5:tools/hipvs_bridge/seekdb_cuvs_bridge.c" > "$WORK/base_bridge.c")
(cd "$REPO" && git show "$V5:tools/hipvs_bridge/seekdb_cuvs_bridge.h" > "$WORK/seekdb_cuvs_bridge.h")
cp "$BRIDGE_DIR/seekdb_cuvs_bridge.c" "$WORK/mine_bridge.c"
cp "$HERE/stub/cuvs_stub.c" "$WORK/cuvs_stub.c"
cp "$HERE/driver.c" "$WORK/driver.c"
wc -l "$WORK/base_bridge.c" "$WORK/mine_bridge.c"

echo "=== 2. 分别编译（基线 / 本实现的 baseline 预设）==="
gcc -std=gnu17 -O1 -Wall -D__HIP_PLATFORM_AMD__ -I"$HERE/stub" -I"$WORK" \
    "$WORK/base_bridge.c" "$WORK/cuvs_stub.c" "$WORK/driver.c" -o "$WORK/base_drive" -lpthread
gcc -std=gnu17 -O1 -Wall -D__HIP_PLATFORM_AMD__ -I"$HERE/stub" -I"$WORK" \
    "$WORK/mine_bridge.c" "$WORK/cuvs_stub.c" "$WORK/driver.c" -o "$WORK/mine_drive" -lpthread

echo "=== 3. 运行并对比 ==="
"$WORK/base_drive" > "$WORK/base.out"
SEEKDB_CUVS_PRESET=baseline "$WORK/mine_drive" > "$WORK/mine.out"
echo "--- 冻结基线 ---"; cat "$WORK/base.out"
echo "--- 本实现(baseline 预设) ---"; cat "$WORK/mine.out"
if diff -u "$WORK/base.out" "$WORK/mine.out" > "$WORK/diff.out"; then
  echo "RESULT: PASS —— 两边的建图参数、检索参数与结果形状完全一致"
  exit 0
fi
echo "RESULT: FAIL —— 两边不一致："
cat "$WORK/diff.out"
exit 1
