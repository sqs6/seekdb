#!/usr/bin/env bash
# 校验「基线自带的验收脚本」依赖的运行时日志/trace 关键字，我们的改动没有删掉它们。
#
# 为什么需要：官方硬门槛第 5 条「基线已有的基本索引功能仍可使用」，
# 而基线自带的 docs/gpu-vector-index-hipvs-cuvs/validation/run_ddl_persistence_smoke.sh
# 正是靠这些关键字来判定的（重启后走 GPU、重启后不重建、日志里有快照记录…）。
# 我们改了快照格式（OBCUVS02 -> OBCUVS03）与写入顺序，这些关键字必须还在。
#
# 用法：bash tools/hipvs_bridge/test/run_smoke_compat.sh
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
ADAPTOR="$REPO/src/oblib/lib/vector/ob_vsag_adaptor.cpp"
SMOKE="$REPO/docs/gpu-vector-index-hipvs-cuvs/validation"

FAIL=0

# 1) 冒烟脚本里用到的关键字，必须能在源码里找到对应的日志/trace 语句
need_in_src=(
  "serialized GPU CAGRA snapshot"   # 快照写入日志
  "deserialized GPU CAGRA snapshot" # 快照读取日志
  "K(built)"                       # built=true
  "K(graph_bytes)"                 # graph_bytes=NNN
  "cuvs_serve"                     # GPU 侧提供查询的 trace
  "cuvs_build"                     # GPU 侧建图的 trace
  "fdeserialize"                   # VSAG 侧从文件反序列化的 trace
)
echo "=== 1. 冒烟脚本依赖的日志/trace 关键字是否仍在源码里 ==="
for pat in "${need_in_src[@]}"; do
  if grep -qF -- "$pat" "$ADAPTOR"; then
    printf '  [ OK ] %s\n' "$pat"
  else
    printf '  [FAIL] 源码里找不到：%s\n' "$pat"
    FAIL=1
  fi
done

# 2) 冒烟脚本断言「不应出现」的运行时关键字：源码里对应的分支必须仍然可达
#    （lazily built VSAG fallback 是 VSAG 回退告警；GPU 路径正常时不该出现）
echo
echo "=== 2. 冒烟脚本断言不应出现的关键字（源码里对应分支仍在，运行时不该触发）==="
for pat in "lazily built VSAG fallback"; do
  if grep -qF -- "$pat" "$ADAPTOR"; then
    printf '  [ OK ] %s（该告警分支仍在源码中，GPU 路径正常时不会触发）\n' "$pat"
  else
    printf '  [FAIL] 源码里找不到：%s\n' "$pat"
    FAIL=1
  fi
done

# 3) 冒烟脚本会跑的两个脚本本身要存在且语法正确
echo
echo "=== 3. 基线自带验收脚本的存在性与语法 ==="
for s in run_ddl_persistence_smoke.sh run_gpu_smoke.sh; do
  if [ -f "$SMOKE/$s" ]; then
    if bash -n "$SMOKE/$s" 2>/dev/null; then
      printf '  [ OK ] %s\n' "$s"
    else
      printf '  [FAIL] %s 语法有问题\n' "$s"
      FAIL=1
    fi
  else
    printf '  [FAIL] 找不到 %s\n' "$s"
    FAIL=1
  fi
done

# 4) 快照写入端与读取端的**字段顺序**必须一致（我们把 graph_bytes 挪到了 vectors 之后）
echo
echo "=== 4. 快照写入端与读取端的字段顺序一致 ==="
fields_of() {   # $1=函数名 $2=调用名(out/in)
  sed -n "/$1/,/^}/p" "$ADAPTOR" \
    | grep -oF "$2(out, " -e "$2(in, " 2>/dev/null >/dev/null
  sed -n "/$1/,/^}/p" "$ADAPTOR" \
    | grep -oE "ob_cuvs_$2\((out|in), [^,]*," \
    | sed -E "s/.*ob_cuvs_$2\((out|in), //; s/,$//" \
    | sed -E "s/&//g; s/OB_CUVS_SNAPSHOT_MAGIC/magic/; s/dim32/dim/; s/reserved/reserved/; s/n64/n/; s/ent->buf_ids_\.data\(\)/ids/; s/ent->buf_vecs_\.data\(\)/vecs/; s/graph_bytes/graph_bytes/"
}
write_order=$(fields_of "static bool ob_cuvs_serialize" "write")
read_order=$(fields_of "static int ob_cuvs_deserialize" "read")
echo "  写入端顺序: $(echo $write_order | tr '\n' ' ')"
echo "  读取端顺序: $(echo $read_order | tr '\n' ' ')"
# 读取端比写入端少第一个字段（magic 由调用方先读掉）
w_first=$(echo "$write_order" | head -1)
w_rest=$(echo "$write_order" | tail -n +2)
if [ "$w_first" = "magic" ] && [ "$w_rest" = "$read_order" ]; then
  printf '  [ OK ] 字段顺序一致（magic 由调用方先读掉）\n'
else
  printf '  [FAIL] 字段顺序不一致\n'
  FAIL=1
fi

echo
if [ "$FAIL" = "0" ]; then
  echo "RESULT: PASS —— 基线验收脚本依赖的运行时关键字与快照读写两端都保持兼容"
else
  echo "RESULT: FAIL"
fi
exit "$FAIL"
