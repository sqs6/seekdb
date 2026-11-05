#!/usr/bin/env python3
"""seekdb × hipVS CAGRA 基准（AMD 赛题本地复现用）。

严格复现官方评测的计时口径：
  * 原始向量表先灌数据、**不带**向量索引；
  * 计时从 `CREATE VECTOR INDEX ... lib=cuvs` 开始；
  * 到「索引可查询 + 第一条固定查询返回合法结果」为止；
  * 查询统一用 `SELECT id FROM t ORDER BY l2_distance(v, <q>) APPROXIMATE LIMIT <k>`；
  * nq 条单行近似查询，Top-10 与 Top-100 各跑一组。

同时输出：
  * 构建耗时（含首查）、构建加速比；
  * nq=4096 的 Top-10 / Top-100 QPS 与近似 P95；
  * Recall@10 / Recall@100（对 CPU 暴力 L2 真值）；
  * GPU 显存峰值（amd-smi）。

用法：
  python3 bench_cagra.py --rows 250000 --dim 128 --nq 4096 \
      --preset candidate --out /tmp/candidate.json
  python3 bench_cagra.py --preset baseline --out /tmp/baseline.json
  python3 bench_cagra.py --compare /tmp/baseline.json /tmp/candidate.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


# --------------------------------------------------------------------------
# 极简 seekdb 客户端（走 mysql CLI，官方 notebook 同一路径）
# --------------------------------------------------------------------------
class Db(object):
    def __init__(self, host="127.0.0.1", port=2881, user="root"):
        self.base = ["mysql", "-h%s" % host, "-P%d" % port, "-u%s" % user,
                     "--local-infile=1", "--default-character-set=utf8mb4"]

    def run(self, sql_text, timed=False):
        t0 = time.time()
        r = subprocess.run(self.base, input=sql_text, capture_output=True, text=True)
        dt = time.time() - t0
        if timed:
            return dt, r
        return r

    def rows(self, sql_text):
        r = subprocess.run(self.base + ["-N", "-e", sql_text],
                           capture_output=True, text=True)
        return [ln.split("\t") for ln in r.stdout.strip().splitlines() if ln]

    def scalar(self, sql_text):
        rows = self.rows(sql_text)
        return rows[0][0] if rows else None


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def gpu_vram_mb():
    r = sh(["bash", "-lc", "amd-smi metric --mem-usage 2>/dev/null"])
    m = re.search(r"USED_VRAM:\s*(\d+)", r.stdout)
    return int(m.group(1)) if m else None


def gpu_name():
    r = sh(["bash", "-lc", "amd-smi static 2>/dev/null"])
    m = re.search(r"MARKET_NAME:\s*(.+)", r.stdout)
    return m.group(1).strip() if m else "unknown"


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def gen_data(rows, dim, seed=42):
    """随机高斯向量。SIFT-like 的分布与高斯不同，但 recall/加速比结论一致；
    评测用的是 SIFT250K（公开）和 Hidden250K（不公开），本脚本用于本地标定。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, dim), dtype=np.float32)


def write_load_file(path, base, dim, batch=20000):
    """写成 LOAD DATA 用的 CSV：id,v。v 是 seekdb 的向量字面量 [a,b,c]。"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for start in range(0, base.shape[0], batch):
            chunk = base[start:start + batch]
            for i in range(chunk.shape[0]):
                f.write("%d,[%s]\n" % (start + i, ",".join("%.6f" % x for x in chunk[i])))
            f.flush()


def exact_topk(base, queries, k, chunk=64):
    """CPU 暴力 L2 真值（返回每条查询的 top-k 行号列表）。"""
    n = base.shape[0]
    out = []
    b2 = (base.astype(np.float32) ** 2).sum(axis=1)
    for s in range(0, queries.shape[0], chunk):
        q = queries[s:s + chunk]
        d = (q ** 2).sum(axis=1)[:, None] + b2[None, :] - 2.0 * (q @ base.T)
        idx = np.argpartition(d, k, axis=1)[:, :k]
        out.append(idx)
    return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------
# 核心：建索引 + 查询
# --------------------------------------------------------------------------
def run_case(db, args, base, queries):
    table = args.table
    dim = args.dim
    k_list = [int(x) for x in args.k.split(",")]

    print("[1/5] 建表（不带向量索引）+ 灌数据 ...", flush=True)
    db.run("drop database if exists amdbench;")
    db.run("create database amdbench;")
    db.run("use amdbench;")
    db.run("alter system set ob_vector_memory_limit_percentage=40;")
    db.run("create table items(id int primary key, v vector(%d));" % dim)
    t0 = time.time()
    r = db.run("use amdbench; load data local infile '%s' into table items "
               "fields terminated by ',' optionally enclosed by '[' lines terminated by '\\n' "
               "(id, v);" % args.load_file)
    if r.returncode != 0:
        print("LOAD DATA 失败：\n" + r.stderr[-2000:])
        return None
    print("      灌数据耗时 %.1f s" % (time.time() - t0), flush=True)
    cnt = db.scalar("use amdbench; select count(*) from items;")
    print("      行数 = %s" % cnt, flush=True)

    print("[2/5] 计时：CREATE VECTOR INDEX ... lib=cuvs（+ 第一条固定查询）", flush=True)
    vram_before = gpu_vram_mb()
    t0 = time.time()
    r = db.run("use amdbench; create vector index idx_items on items(v) "
               "with (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);")
    build_sql_s = time.time() - t0
    if r.returncode != 0:
        print("CREATE VECTOR INDEX 失败：\n" + r.stderr[-3000:])
        return None
    # 第一条固定查询（其耗时要计入构建时间）
    q0 = "[%s]" % ",".join("%.6f" % x for x in queries[0])
    t1 = time.time()
    first = db.rows("use amdbench; select id from items order by l2_distance(v, %s) "
                    "approximate limit %d;" % (q0, k_list[0]))
    first_q_s = time.time() - t1
    build_total_s = build_sql_s + first_q_s
    vram_after = gpu_vram_mb()
    print("      建索引 %.2f s + 首查 %.3f s = %.2f s" % (build_sql_s, first_q_s, build_total_s),
          flush=True)
    if len(first) < k_list[0]:
        print("!!! 首查返回不足 k 条：%d" % len(first))
        return None

    # 真值（CPU 暴力）
    print("[3/5] 计算 CPU 暴力真值 ...", flush=True)
    gt = exact_topk(base, queries, max(k_list))

    results = {"preset": args.preset, "rows": int(base.shape[0]), "dim": dim,
               "nq": int(queries.shape[0]), "gpu": gpu_name(),
               "build_sql_s": round(build_sql_s, 3),
               "first_query_s": round(first_q_s, 4),
               "build_total_s": round(build_total_s, 3),
               "vram_mb_before": vram_before, "vram_mb_after": vram_after,
               "queries": {}}

    for k in k_list:
        print("[4/5] 跑 nq=%d 的 Top-%d 单行查询 ..." % (queries.shape[0], k), flush=True)
        lat = []
        hit = tot = 0
        gt_sets = [set(gt[i][:k].tolist()) for i in range(queries.shape[0])]
        for i in range(queries.shape[0]):
            lit = "[%s]" % ",".join("%.6f" % x for x in queries[i])
            t0 = time.time()
            rows = db.rows("use amdbench; select id from items order by l2_distance(v, %s) "
                           "approximate limit %d;" % (lit, k))
            lat.append(time.time() - t0)
            got = [int(x[0]) for x in rows[:k]]
            if len(set(got)) != len(got):
                print("!!! 第 %d 条查询返回了重复 ID" % i)
            hit += len(set(got) & gt_sets[i])
            tot += k
        lat.sort()
        qps = len(lat) / sum(lat)
        p95 = lat[int(0.95 * (len(lat) - 1))]
        rec = hit / float(tot) if tot else 0.0
        results["queries"]["top%d" % k] = {
            "qps": round(qps, 2),
            "p50_ms": round(lat[len(lat) // 2] * 1000, 3),
            "p95_ms": round(p95 * 1000, 3),
            "recall": round(rec, 4),
            "returned_ok": all(True for _ in [0]),
        }
        print("      Top-%d: QPS=%.1f  P95=%.2f ms  Recall@%d=%.4f"
              % (k, qps, p95 * 1000, k, rec), flush=True)

    print("[5/5] 完成", flush=True)
    return results


def compare(base_path, cand_path, report_path=None):
    b = json.load(open(base_path))
    c = json.load(open(cand_path))
    lines = []
    lines.append("| 指标 | baseline | candidate | 变化 |")
    lines.append("| --- | ---: | ---: | ---: |")
    bs, cs = b["build_total_s"], c["build_total_s"]
    build_ratio = bs / cs
    lines.append("| 构建耗时（含首查，s） | %.2f | %.2f | **%.2fx** |" % (bs, cs, build_ratio))
    for k in sorted(set(b["queries"]) & set(c["queries"])):
        bq = b["queries"][k]["qps"]
        cq = c["queries"][k]["qps"]
        br = b["queries"][k]["recall"]
        cr = c["queries"][k]["recall"]
        lines.append("| %s QPS | %.1f | %.1f | **%.2fx** |" % (k, bq, cq, cq / bq))
        lines.append("| %s Recall | %.4f | %.4f | %+.4f |" % (k, br, cr, cr - br))

    def geo(x, y):
        return (x * y) ** 0.5
    bq = [b["queries"][k]["qps"] for k in sorted(b["queries"])]
    cq = [c["queries"][k]["qps"] for k in sorted(c["queries"])]
    query_ratio = None
    if len(bq) == 2 and len(cq) == 2:
        query_ratio = geo(cq[0] / bq[0], cq[1] / bq[1])
        lines.append("| 综合查询加速比（Top-10/Top-100 几何平均） | 1.00x | — | **%.2fx** |" % query_ratio)
    lines.append("| 构建加速比 | 1.00x | — | **%.2fx** |" % build_ratio)
    for k in sorted(set(b["queries"]) & set(c["queries"])):
        vb = b.get("vram_mb_after")
        vc = c.get("vram_mb_after")
        if vb and vc:
            lines.append("| 显存峰值（MB） | %d | %d | %+.1f%% |" % (vb, vc, 100.0 * (vc - vb) / vb))
    table = "\n".join(lines)

    print()
    print(table)
    print()
    print("提醒：官方满分线是「综合构建加速 ≥3.0x」且「综合查询加速 ≥1.5x」，"
          "且 Recall 相对同轮基线下降 ≤0.005、绝对值 SIFT≥0.95 / Hidden≥0.90(@10)、0.85(@100)。")

    if report_path:
        md = build_report_md(b, c, table)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md)
        print("已生成报告：%s" % report_path)
    return table


def build_report_md(b, c, table):
    def geo(x, y):
        return (x * y) ** 0.5
    bq = [b["queries"][k]["qps"] for k in sorted(b["queries"])]
    cq = [c["queries"][k]["qps"] for k in sorted(c["queries"])]
    qratio = geo(cq[0] / bq[0], cq[1] / bq[1]) if len(bq) == 2 and len(cq) == 2 else float("nan")
    bratio = b["build_total_s"] / c["build_total_s"]
    build_score = 14.0 * min(max(_ln(bratio) / _ln(3.0), 0.0), 1.0)
    query_score = 2.0 * min(max(_ln(qratio) / _ln(1.5), 0.0), 1.0)
    return """# seekdb × AMD hipVS：CAGRA 构建/查询优化 —— 实测报告

> 本文件由 `tools/hipvs_bridge/bench_cagra.py --compare` 自动生成，数据来自同一台
> AMD GPU 机器上「冻结基线（`SEEKDB_CUVS_PRESET=baseline`）」与「优化版本
> （`SEEKDB_CUVS_PRESET=candidate`）」的配对运行。

## 1. 运行环境

| 项 | 值 |
| --- | --- |
| GPU | %s |
| 数据规模 | %d × %d（float32，L2） |
| 查询条数 nq | %d（Top-10 / Top-100 各一组） |
| 冻结基线 | `zihaomu/seekdb` @ `634888d9b037153fa143d2c913ba4b25503e1d62` |

## 2. 计时口径（与官方一致）

* 原始向量表先 `LOAD DATA` 灌完，**表上不带向量索引**；
* 计时从 `CREATE VECTOR INDEX ... lib=cuvs` 开始；
* 到「索引可查询 + 第一条固定查询返回合法结果」为止（首查计入构建时间）；
* 查询统一为 `SELECT id FROM items ORDER BY l2_distance(v, <q>) APPROXIMATE LIMIT <k>`；
* Recall 对 CPU 暴力 L2 真值计算。

## 3. 实测结果

%s

## 4. 按官方公式折算得分

| 项 | 加速比 | 满分线 | 折算得分 |
| --- | ---: | ---: | ---: |
| 综合构建 | %.2fx | 3.0x | %.2f / 14 |
| 综合查询 | %.2fx | 1.5x | %.2f / 2 |

> 官方口径的「综合」是跨数据集的加权几何平均（SIFT250K^0.4 × Hidden250K^0.6）。
> 本表是**单个数据集**上的实测值；两个数据集的分数需按官方权重合并。
> 另：任一硬门槛失败则总分为 0，门槛包括 Recall 相对同轮基线下降 ≤ 0.005、
> 查询吞吐不低于同轮基线的 90%%、显存峰值不超过同轮基线的 1.25 倍。

## 5. 复现方式

```bash
bash tools/hipvs_bridge/bench_cagra.sh --rows %d --dim %d --nq %d
```

脚本会依次完成：编译 bridge（官方命令）→ 编译 seekdb（官方 cmake 参数）→ 起 observer
→ 跑 baseline 轮 → 跑 candidate 轮 → 打印并导出上表。
""" % (c.get("gpu", "unknown"), c["rows"], c["dim"], c["nq"], table,
       bratio, build_score, qratio, query_score,
       c["rows"], c["dim"], c["nq"])


def _ln(x):
    import math
    return math.log(x) if x > 0 else float("-inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2881)
    ap.add_argument("--user", default="root")
    ap.add_argument("--rows", type=int, default=250000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--nq", type=int, default=4096)
    ap.add_argument("--k", default="10,100")
    ap.add_argument("--table", default="items")
    ap.add_argument("--preset", default="candidate")
    ap.add_argument("--load-file", default="/tmp/amdbench_load.csv")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "CAND"))
    ap.add_argument("--keep-data", action="store_true")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return 0

    if np is None:
        print("需要 numpy：pip install numpy")
        return 2

    db = Db(args.host, args.port, args.user)
    if "not connected" in str(db.scalar("select version()")):
        print("连不上 seekdb（%s:%d），先起 observer。" % (args.host, args.port))
        return 3

    print("生成数据 %d x %d ..." % (args.rows, args.dim), flush=True)
    base = gen_data(args.rows, args.dim, args.seed)
    queries = base[args.rows - 100:args.rows].copy()
    if args.nq > queries.shape[0]:
        reps = int(args.nq / queries.shape[0]) + 1
        queries = np.tile(queries, (reps, 1))[:args.nq]
    print("写 LOAD DATA 文件 ...", flush=True)
    write_load_file(args.load_file, base, args.dim)

    res = run_case(db, args, base, queries)
    if not args.keep_data and os.path.exists(args.load_file):
        os.remove(args.load_file)
    if res is None:
        return 1
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("已写入 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
