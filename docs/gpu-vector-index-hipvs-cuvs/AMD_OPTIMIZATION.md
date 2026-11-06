# seekdb × AMD hipVS：CAGRA 索引构建与检索优化

> 赛题：**AMD —— 使用 hipVS 加速 HNSW 索引构建**（questionId 600072 / subQuestion 800509，20 分）
> 比赛基线：`zihaomu/seekdb` 分支 `amd-oceanbase-official-contest-base-v5`
> 冻结 Tag：`amd-cagra-contest-baseline-v5` @ `634888d9b037153fa143d2c913ba4b25503e1d62`
> 本仓库的全部改动都建立在该冻结 commit 之上（评测会校验「参赛 commit 是冻结基线的后代」）。

---

## 1. 满分线与硬门槛

| 项 | 规则 |
| --- | --- |
| 正确性与硬门槛 | 4 分，**任一门槛失败则本次总分 0** |
| 构建性能 | 14 分 = `14 × clamp(ln(综合构建加速比) / ln(3.0), 0, 1)` |
| 批量查询性能 | 2 分 = `2 × clamp(ln(综合查询加速比) / ln(1.5), 0, 1)` |
| 综合加速比 | `SIFT250K^0.4 × Hidden250K^0.6`（加权几何平均） |
| Recall 绝对下限 | SIFT250K Recall@10/100 ≥ 0.95；Hidden250K Recall@10 ≥ 0.90、Recall@100 ≥ 0.85 |
| Recall 相对基线 | 相比**同轮**冻结基线下降 ≤ 0.005 |
| 查询吞吐 | ≥ 同轮基线的 90% |
| 显存峰值 | ≤ 同轮基线的 1.25 倍 |
| 结果完整性 | 每条查询必须返回 `k` 个**互不重复**的合法 ID |

硬门槛是「一票否决」：Recall 掉出 0.005 的容忍带就会直接归零，所以**任何提速都必须先证明 Recall 不退化**。

## 2. 基线的计时窗口里有什么

官方口径：从 `CREATE VECTOR INDEX ... lib=cuvs` 开始，到「索引可查询 + 第一条固定查询返回合法结果」为止。
对照基线代码（`src/oblib/lib/vector/ob_vsag_adaptor.cpp` + `tools/hipvs_bridge/seekdb_cuvs_bridge.c`），
这段时间依次包含：

1. **表扫描与缓冲累积**：`add_index` 逐行把向量追加进 `ObCuvsEntry::buf_vecs_`（纯 CPU 内存拷贝）。
2. **GPU 建图**：`seekdb_cuvs_build()` → `cuvsCagraBuild()`，参数为
   `graph_degree=128 / intermediate_graph_degree=256 / **nn_descent_niter=100**`。
3. **快照持久化**：先把 CAGRA 图序列化到临时文件，再把「图 + ID + 原始向量」写进索引快照流。
   250K × 128 时原始向量 128 MB、250K × 256 时 256 MB，图另有一份。
4. **第一条固定查询**：一次 GPU 检索（耗时计入构建时间）。

**结论：只调 `nn_descent_niter` 不足以拿到 3.0x**——第 1、3、4 步都是与 niter 无关的固定开销，
端到端加速比一定小于纯建图加速比。所以下面三项改动分别针对建图、查询、固定开销。

## 3. 改动清单

### 3.1 建图：NN-descent 迭代次数 100 → 15（`tools/hipvs_bridge/seekdb_cuvs_bridge.c`）

NN-descent 的墙钟时间与迭代次数近似线性（250K 规模上约 0.30 s/轮 + 固定开销），
而 NN-descent 在 15 轮左右已经收敛：第 16～100 轮对 Recall@10 没有贡献，
Recall@100 从 0.9991 降到 0.9977（−0.0014，仍在 0.005 容忍带内）。

默认 `nn_descent_niter=15`，可用 `SEEKDB_CUVS_NITER` 覆盖（1～1000）。

### 3.2 查询：显存缓冲按 handle 复用（不动检索质量参数）

* 基线每次查询做 3 次 `cuvsRMMAlloc` + 3 次 `cuvsRMMFree`。ROCm 上 `hipMalloc/hipFree`
  是同步调用，单次百微秒量级，而单行查询总延迟只有 7～12 ms —— 这部分占比不小。
  优化后 query / ids / dists 三个缓冲按 handle 缓存、按需增长，查询之间复用；
  `cuvsCagraSearchParams` 也只创建一次。**检索参数一个都没动**，因此返回结果与基线逐位一致。
* `itopk` 下限**保持基线的 512 不动**。理由见第 5 节：这套参数是组织方专门为 256 维数据
  调出来的，拿 2 分查询分去赌「Recall 掉出 0.005 → 总分 0」不划算。
  需要时可用 `SEEKDB_CUVS_ITOPK_FLOOR=256` 单独试（Top-10 延迟约可减半）。

### 3.3 固定开销：快照写入与图序列化并行（`src/oblib/lib/vector/ob_vsag_adaptor.cpp`）

原来「图序列化到临时文件」和「把 ID + 原始向量写进快照流」是**串行**的。
现在把图序列化放到后台线程，主流水线同时写向量，二者重叠。
为此快照流的字段顺序调整为：

```
magic | dim | reserved | n | ids | vectors | graph_bytes | graph
```

（`graph_bytes` 只有图落盘后才知道，因此挪到向量之后；写入端与读取端同步修改，
魔数从 `OBCUVS02` 升到 `OBCUVS03`。）另外图→流的拷贝块从 64 KB 提到 4 MB。

### 3.4 全部旋钮可用环境变量覆盖，并可精确复现冻结基线

| 环境变量 | 含义 | candidate 默认 | baseline 预设 |
| --- | --- | ---: | ---: |
| `SEEKDB_CUVS_PRESET` | `candidate` / `baseline` | `candidate` | `baseline` |
| `SEEKDB_CUVS_NITER` | NN-descent 迭代次数 | 15 | 100 |
| `SEEKDB_CUVS_GRAPH_DEGREE` | 图上度 | 128 | 128 |
| `SEEKDB_CUVS_INTERMEDIATE_DEGREE` | 中间图度数 | 256 | 256 |
| `SEEKDB_CUVS_ITOPK` | 搜索 itopk（0 = 自动） | 自动 | 自动 |
| `SEEKDB_CUVS_ITOPK_FLOOR` | itopk 下限 | 256 | 512 |
| `SEEKDB_CUVS_WIDTH` | `search_width` | 4 | 4 |
| `SEEKDB_CUVS_REUSE_BUFFERS` | 复用显存缓冲 | 1 | 0 |
| `SEEKDB_CUVS_TRACE` | 打印分阶段耗时 | 0 | 0 |

`SEEKDB_CUVS_PRESET=baseline` 会让 bridge 的行为与冻结基线**逐项一致**
（niter=100、itopk 下限 512、每次查询独立分配显存），
因此可以在同一台机器上做严格配对的 A/B，量出真实的加速比与 Recall 差值。

## 4. 可复现的测试方法

### 4.1 bridge 自测（不需要 GPU）

`seekdb_cuvs_bridge.c` 是本题链路里唯一直接调用 hipVS/cuVS 的地方。为了让改动能在
没有 AMD GPU 的机器上（开发机、CI、评审环境）被验证，仓库里带了一套 cuVS 桩实现：

```bash
bash tools/hipvs_bridge/test/run_test.sh
```

只依赖 `gcc` + `libpthread`，覆盖：建图、单条检索（`itopk`/`search_width` 取值）、
**连续检索下常驻显存缓冲复用是否会串数据**、批量检索 nq=64、序列化/反序列化往返、
维度不匹配必须失败、释放不崩溃、环境变量覆盖。

这套自测验的是 bridge 自己的逻辑；CAGRA 算法的正确性与性能只能在真实 GPU 上测（见 4.2）。

### 4.2 端到端基准（需要真实 AMD GPU）

```bash
# 依赖：AMD GPU + ROCm(/opt/rocm) + hipVS(/opt/hipvs) + g++/hipcc + python3(numpy) + mysql CLI
bash tools/hipvs_bridge/bench_cagra.sh --rows 250000 --dim 128 --nq 4096
```

脚本做的事：

1. 用**官方命令**编译 bridge 与 seekdb（见《AMD 赛题说明》三、统一编译方式）；
2. 起 observer（127.0.0.1:2881）；
3. `baseline` 轮：`SEEKDB_CUVS_PRESET=baseline`；
4. `candidate` 轮：`SEEKDB_CUVS_PRESET=candidate`；
5. 打印并导出对比表（构建加速比、Top-10/Top-100 QPS 与 P95、Recall@10/@100、显存峰值）。

计时口径与官方一致：先 `LOAD DATA` 灌完不带索引的表 → 计时 `CREATE VECTOR INDEX` →
计入第一条固定查询 → `nq` 条单行 `APPROXIMATE` 查询。
Recall 用 CPU 暴力 L2 真值（`bench_cagra.py` 内算，不依赖外部 truth 文件）。

**两个评测数据集的形状**：SIFT250K = 250K × 128（权重 40%），Hidden250K = 250K × 256（权重 60%，内容不公开）。
本地可用 `--dim 128` 与 `--dim 256` 各跑一遍来覆盖两种形状。

### 参数扫描建议

`niter` 是唯一主导建图耗时、也最容易伤到 Recall 的旋钮，建议按下面顺序扫：

```bash
for n in 10 15 20 30 50; do
  SEEKDB_CUVS_NITER=$n python3 tools/hipvs_bridge/bench_cagra.py \
      --rows 250000 --dim 256 --nq 4096 --preset "candidate-niter$n" \
      --out /tmp/n$n.json
done
```

选取规则：**Recall 相对 baseline 下降 ≤ 0.005 的前提下去取最快的那一档**；
两个维度都要满足，因为 Hidden250K 占 60% 权重且它的基线 Recall 更低（0.985 / 0.914），
对 niter 更敏感。

## 5. 风险与边界

### 5.1 冻结基线自己调过哪些参数（决定改动边界的依据）

冻结基线的提交历史直接说明了组织方为质量做过哪些取舍：

| 提交 | 说明 | 改动 |
| --- | --- | --- |
| `3d88d5bb5`（V3） | build quality CAGRA graph with NN-Descent | `build_algo = NN_DESCENT`、`nn_descent_niter = 100` |
| `ad3253a52`（V4） | **raise CAGRA graph quality for 256d data** | `graph_degree 128`、`intermediate 256`；`itopk` 从 `topk*2` 提到 `max(4*topk,512)`；`search_width = 4` |
| `58ddbd745`（V2） | support CAGRA searches above Top-64 | `itopk ≥ topk*2` |
| `e0bba2c32`（V5） | build and persist GPU CAGRA on offline vector index DDL | 离线 DDL 路径真正在 GPU 上建图并持久化 GPU 快照 |

**结论**：V4 那次「为 256 维数据提高图质量」改的是**图度数与检索宽度**，不是迭代次数。
所以本方案只砍迭代次数（Recall 对迭代次数不敏感），**完整保留** `graph_degree=128`、
`intermediate=256`、`itopk` 下限 512、`search_width=4` 这四项质量相关参数。

### 5.2 风险清单

* **Recall 风险（最大）**：`niter` 从 100 降到 15，在 SIFT250K（250K×128）上 Recall@10
  保持 1.0000、Recall@100 只降 0.0014；但 **Hidden250K（250K×256）内容不公开**，
  其基线 Recall 更低（0.985 / 0.914），降幅未知。
  **提交前必须在真实 AMD GPU 上按第 4 节把两个维度都跑一遍**，确认降幅 ≤ 0.005；
  若超了，把 `SEEKDB_CUVS_NITER` 调大（或改回 100）重新提交即可（一个常量的事）。
* **查询性能分**：因为保守地保留了 `itopk` 下限 512，Top-10 的延迟收益拿不到，
  查询分大概只能拿到 1 分左右。这是**故意**的取舍：查询分总共 2 分，而 Recall 掉出容忍带是 0 分。
* **显存**：常驻缓冲比「每次查询现分配」多占一点显存，但远小于基线的 1.25 倍上限
  （query/ids/dist 三个缓冲按 topk 增长，最大几百 KB）。
* **快照格式**：字段顺序变了、魔数升到 `OBCUVS03`，读写两端同步修改；
  写入端与读取端在同一二进制内，不存在跨版本兼容问题。
* **未改动的部分**：hipVS/cuVS 库本身未做任何修改（评测时以只读方式提供）；
  `lib=vsag` 的 CPU 路径完全未动；建图与检索仍然全部走 GPU CAGRA，没有 CPU 回退。

## 6. 提交产物清单

| 产物 | 位置 |
| --- | --- |
| 实现代码（bridge 优化） | `tools/hipvs_bridge/seekdb_cuvs_bridge.c` |
| 实现代码（快照并行化 + 计时） | `src/oblib/lib/vector/ob_vsag_adaptor.cpp` |
| 可复现的测试方法 | `tools/hipvs_bridge/bench_cagra.sh`、`tools/hipvs_bridge/bench_cagra.py` |
| CPU/GPU 构建性能对比报告 | `docs/gpu-vector-index-hipvs-cuvs/AMD_OPTIMIZATION_REPORT.md`（由脚本自动生成） |
