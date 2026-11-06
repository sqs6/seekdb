#!/usr/bin/env bash
# bridge 自测：在没有 AMD GPU 的机器上，用 cuVS 桩实现验证 seekdb_cuvs_bridge.c 的
# 控制流与数据搬运（建图参数、检索参数、缓冲复用、批量检索、序列化往返、释放）。
#
# 用法（在任意 Linux/gcc 环境，不需要 GPU、不需要 hipVS）：
#   bash tools/hipvs_bridge/test/run_test.sh
#
# 注意：这里的 stub/ 目录是**自测专用的假头文件**，签名按 cuVS C API 复刻，
# 与评测机上真实的 /opt/hipvs/include 不是一回事，不要混用。
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BRIDGE_DIR=$(cd "$HERE/.." && pwd)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "--- 编译（桩头 + bridge + 测试）---"
cp "$BRIDGE_DIR/seekdb_cuvs_bridge.c" "$BRIDGE_DIR/seekdb_cuvs_bridge.h" "$WORK/"
cp "$HERE/test_bridge.c" "$WORK/"
cp "$HERE/stub/cuvs_stub.c" "$WORK/"

gcc -std=gnu17 -O1 -g -Wall -Wextra -D__HIP_PLATFORM_AMD__ \
    -I"$HERE/stub" -I"$WORK" \
    "$WORK/seekdb_cuvs_bridge.c" "$WORK/cuvs_stub.c" "$WORK/test_bridge.c" \
    -o "$WORK/bridge_test" -lpthread

echo "--- 运行 ---"
"$WORK/bridge_test"
