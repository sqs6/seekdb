# seekdb × AMD hipVS：CAGRA 构建性能对比报告

> 本文件是《AMD 赛题说明》要求的提交产物之一：**「可复现的测试方法及 CPU/GPU 构建性能对比报告」**。
> 改动依据、计分口径与风险分析见同目录 `AMD_OPTIMIZATION.md`；本文件只放**实测数据**。
>
> 数字不是手填的：由 `tools/hipvs_bridge/bench_cagra.py --compare` 在同一台 AMD GPU 机器上
> 配对跑「冻结基线」与「参赛版本」后自动导出，命令见第 3 节。

---

## 1. 实测环境

| 项 | 值 |
| --- | --- |
| GPU | （由脚本填入，amd-smi MARKET_NAME） |
| ROCm / hipVS | （填入） |
| 评测机 CPU / 内存 | （填入） |
| 数据集 | SIFT250K 形状 = 250000 × 128；Hidden250K 形状 = 250000 × 256 |
| 索引参数 | `distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64`（官方固定） |
| 查询 | nq=4096 条单行 `APPROXIMATE` 查询，Top-10 / Top-100 各一组 |

## 2. 结果

<!-- 下面这张表由 bench_cagra.py --compare 导出后原样贴入；两轮之间的顺序对调再跑一遍取稳定值 -->

| 指标 | 冻结基线（`SEEKDB_CUVS_PRESET=baseline`） | 参赛版本（`SEEKDB_CUVS_PRESET=candidate`） | 变化 |
| --- | ---: | ---: | ---: |
| 构建耗时（含首查，s） | | | |
| Top-10 QPS | | | |
| Top-10 P95 (ms) | | | |
| Top-100 QPS | | | |
| Top-100 P95 (ms) | | | |
| Recall@10 | | | |
| Recall@100 | | | |
| 显存峰值 (MB) | | | |
| CPU 路径 (lib=vsag) 构建耗时（含首查，s） | | | GPU 快 __x |
| Top-10 查询 P50（ms） | （CPU） | （GPU） | GPU 快 __x |
| **综合构建加速比** | 1.00x | — | |
| **综合查询加速比** | 1.00x | — | |

> 「CPU 路径」那两行是**同一份数据、同一组索引参数**下 `lib=vsag`（CPU HNSW）的对照，
> 由 `bench_cagra.py --with-cpu` 测出，用来给出官方要求的「CPU/GPU 构建性能对比」。

### 与官方门槛的对照

| 门槛 | 要求 | 实测 | 结论 |
| --- | --- | --- | --- |
| Recall@10 / @100 绝对下限 | SIFT ≥ 0.95；Hidden ≥ 0.90 / 0.85 | | |
| Recall 相对同轮基线 | 下降 ≤ 0.005 | | |
| 查询吞吐 | ≥ 同轮基线的 90% | | |
| 显存峰值 | ≤ 同轮基线的 1.25 倍 | | |
| 每条查询返回 k 个互不重复 ID | 必须 | | |

### 按官方公式折算

| 项 | 加速比 | 满分线 | 折算得分 |
| --- | ---: | ---: | ---: |
| 综合构建 | | 3.0x | / 14 |
| 综合查询 | | 1.5x | / 2 |

> 官方口径的「综合」是跨数据集加权几何平均：`SIFT250K^0.4 × Hidden250K^0.6`。
> 单个数据集的实测值需按这个权重合并。

## 3. 复现方式

```bash
# 一条命令跑完：编译（官方参数）→ 起 observer → baseline 轮 → candidate 轮 → 导出上表
bash tools/hipvs_bridge/bench_cagra.sh --rows 250000 --dim 128 --nq 4096
bash tools/hipvs_bridge/bench_cagra.sh --rows 250000 --dim 256 --nq 4096
```

脚本内部做的事与官方计时口径一一对应：

1. 用官方命令编译 bridge 与 seekdb
   （`tools/hipvs_bridge/build.sh` + `cmake -DOB_BUILD_CUVS=ON -DCUVS_BRIDGE_LIB=...`）；
2. 先灌完**不带向量索引**的 `items` 表；
3. 计时 `CREATE VECTOR INDEX idx_items ON items(v) WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64)`；
4. **把第一条固定查询的耗时计入构建时间**；
5. 跑 nq=4096 条单行 `APPROXIMATE` 查询，统计 QPS 与近似 P95；
6. 用 CPU 暴力 L2 算真值，给出 Recall@10 / @100；
7. 记录 `amd-smi` 的显存占用。

`baseline` 轮通过 `SEEKDB_CUVS_PRESET=baseline` 让 bridge 精确复现冻结基线
（`nn_descent_niter=100`、每次查询独立分配显存），因此两轮是同机配对，量出的加速比就是官方口径。

### 参数扫描

`niter` 是唯一主导建图耗时、也最容易伤到 Recall 的旋钮。扫描命令：

```bash
for n in 10 15 20 30 50 100; do
  SEEKDB_CUVS_NITER=$n python3 tools/hipvs_bridge/bench_cagra.py \
      --rows 250000 --dim 256 --nq 4096 --preset "niter$n" \
      --out /tmp/niter$n.json
done
```

选取规则：**Recall 相对 baseline 下降 ≤ 0.005 的前提下取最快的一档**，两个维度都要满足。
Hidden250K 占 60% 权重且基线 Recall 更低（0.985 / 0.914），对 niter 比 SIFT 更敏感。

## 4. 与官方参考值的对照

官方《AMD 赛题说明》给出的 V5 冻结基线参考值（评测机为 AMD Radeon PRO W7900D，ROCm 7.2.4）：

| 数据集 | 构建时间（含第一条查询） | Recall@10 / @100 | nq=4096 查询 QPS（Top-10 / Top-100） |
| --- | ---: | ---: | ---: |
| SIFT250K | 20～27 s（中位数约 23 s） | 约 1.000 / 1.000 | 约 147 / 143 |
| Hidden250K | 29～35 s（中位数约 30 s） | 约 0.985 / 0.914 | 约 87 / 85 |

本机绝对时间会与上表不同（机器不同）；**成绩始终与同轮运行的冻结基线配对比较**，
不用上表的绝对值换算。
