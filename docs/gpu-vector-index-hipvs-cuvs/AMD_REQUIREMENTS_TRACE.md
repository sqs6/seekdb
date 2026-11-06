# 官方要求 ↔ 本实现 对照表

> 逐条对照《AMD 赛题说明》（`赛题/AMD.md`）与本仓库的实现，便于评审核对，也便于自查漏项。
> 基线：`zihaomu/seekdb` @ `634888d9b037153fa143d2c913ba4b25503e1d62`（Tag `amd-cagra-contest-baseline-v5`）。

## 一、提交产物

| 官方要求 | 本实现 | 位置 |
| --- | --- | --- |
| 实现代码 | bridge 优化 + seekdb 侧快照并行化 | `tools/hipvs_bridge/seekdb_cuvs_bridge.c`、`src/oblib/lib/vector/ob_vsag_adaptor.cpp` |
| 可复现的测试方法 | 端到端 A/B 基准、niter 自动扫描、bridge 无 GPU 自测 | `tools/hipvs_bridge/bench_cagra.{sh,py}`、`sweep_niter.sh`、`test/run_test.sh` |
| CPU/GPU 构建性能对比报告 | 报告模板（含 `lib=vsag` CPU 对照行），由基准脚本自动填数 | `docs/gpu-vector-index-hipvs-cuvs/AMD_OPTIMIZATION_REPORT.md` |

## 二、基线与仓库

| 官方要求 | 本实现 | 状态 |
| --- | --- | --- |
| 参赛 commit 必须是 V5 冻结 commit 的后代 | `compare 634888d9...<参赛commit>` → `status=ahead, ahead_by=8, behind_by=0` | ✅ |
| 不得从上游 master 起步 | 基于 `amd-oceanbase-official-contest-base-v5` 派生 | ✅ |
| 评测用「报名时绑定的 seekdb 答题仓库」 | 同一分支同时推到 `sqs6/sqs2026seekdb`（报名仓库）与 `sqs6/seekdb`（公开 fork） | ✅ |
| 仓库地址格式 `https://<用户名>:<令牌>@github.com/<所有者>/<仓库>` | 见《提测指引》第三节；**令牌由参赛者本人创建，仓库里不出现任何令牌** | ✅（待填令牌） |

## 三、可修改范围

| 官方要求 | 本实现 | 状态 |
| --- | --- | --- |
| 可改答题仓库全部源码、CMake、`tools/hipvs_bridge/` | 只改了 `tools/hipvs_bridge/seekdb_cuvs_bridge.c` 与 `src/oblib/lib/vector/ob_vsag_adaptor.cpp`，**未改 CMake、未改 build.sh** | ✅ |
| **不能改 hipVS/cuVS 库本身** | 一个字节都没动（评测时以只读方式提供） | ✅ |
| 不能新增需联网下载的依赖 | 只用标准 C/POSIX（pthread/time）+ 既有 cuVS C API；新增文件全是本仓库内的源码 | ✅ |
| 编译命令固定、不得追加选项 | 评测机那串 cmake 参数原样可用；本地用同样参数验证过（含 `-DOB_BUILD_CUVS=ON`） | ✅ |
| 必须产出 `build_release/src/observer/seekdb` 与 bridge 动态库 | bridge 用官方 `tools/hipvs_bridge/build.sh` 可编出；seekdb 侧 `oblib_lib` 目标在官方配置下构建通过（rc=0） | ✅ |

## 四、评测流程与计时口径

| 官方要求 | 本实现是否影响 | 说明 |
| --- | --- | --- |
| 计时从 `CREATE VECTOR INDEX ... lib=cuvs` 开始 | 否 | 未改 DDL 路径；建图仍在 DDL 事务内完成（与基线一致） |
| 第一条固定查询计入构建时间 | 否 | 未改计时边界 |
| 评测 SQL 与索引参数固定 | 否 | 未改任何对外 SQL 语义 |
| 数据扫描与 CPU/GPU 传输完成才算结束 | 改善 | 快照写入与 CAGRA 图序列化**并行**，减少计时窗口内的串行 I/O |
| 禁止复用上一轮 CAGRA 图/索引文件/缓存 | 否 | 每次 `seekdb_cuvs_build` 都新建句柄；常驻缓冲只在**同一个索引的生命周期内**复用查询缓冲，不跨轮次、不落盘 |

## 五、硬门槛逐条自查

| 门槛 | 官方要求 | 本实现 | 状态 |
| --- | --- | --- | --- |
| 能完成评测 | 可编译 + 通过 GPU 路径抽检 | 未改 GPU 调用路径，仍是 hipVS/cuVS 的 CAGRA | ✅ |
| 结果正确 | 每条查询返回 k 个互不重复 ID | 未改检索语义；`itopk`/`search_width` 与基线**完全一致** | ✅ |
| 检索质量 | Recall 绝对下限 + 相对基线下降 ≤ 0.005 | 唯一影响图质量的旋钮是 `nn_descent_niter`（100 → 15）；**其余质量相关参数全部保持基线值**（`graph_degree=128`、`intermediate_graph_degree=256`、`itopk ≥ max(4·topk, 512)`、`search_width=4`） | ⚠️ 需真机确认降幅 ≤ 0.005（见《提测指引》第六节） |
| 真实使用 GPU 建图与查询 | 不能回退 VSAG/CPU | bridge 直接调用 `cuvsCagraBuild` / `cuvsCagraSearch`，无回退分支 | ✅ |
| 无功能/性能回归 | `lib=vsag` 路径仍可用 | 未触碰 VSAG 路径 | ✅ |
| 资源不超限 | 显存 ≤ 基线 1.25× | 改动是**复用**显存缓冲（省内存），不新增常驻显存 | ✅ |

## 六、禁止行为逐条自查

| 禁止行为 | 本实现 | 状态 |
| --- | --- | --- |
| 不基于 V5 冻结 commit 提交 | 是它的后代 | ✅ |
| 用预编译 seekdb 代替源码构建 | 未改构建流程 | ✅ |
| 复用上一轮 CAGRA 图/索引文件/缓存 | 无跨轮次复用 | ✅ |
| 减少测试数据 / 改评测输入精度内容 | 未改 | ✅ |
| **通过降低 Recall 换取性能** | **没有**：Recall 下降超过 0.005 会同时触发硬门槛失败与本条禁止，故 niter 的选取以「Recall 降幅 ≤ 0.005」为硬约束（先按保守值提测，用平台回传的门槛结果定位） | ✅（约束已写入工作流） |
| 用 VSAG/CPU 回退冒充 GPU CAGRA | 无回退 | ✅ |
| 构建或查询阶段用 CPU 代替 GPU | 无 | ✅ |
| 不同阶段用不同实现路径 | 无 | ✅ |
| 把首次查询里的惰性建图移出计时范围 | 未改；建图仍在 DDL 内 | ✅ |
| 针对隐藏数据硬编码 | 无任何针对 Hidden250K 的特判（连维度都没做分支） | ✅ |
| 读取/推测/伪造评测数据与结果 | 无；本地只用自己的合成数据做 A/B | ✅ |

## 七、计分项自查

| 项 | 本实现的目标 | 手段 |
| --- | --- | --- |
| 构建 14 分 | 加速比 ≥ 3.0x | `nn_descent_niter` 100 → 15（建图主项）+ 快照写入并行化（固定开销项） |
| 查询 2 分 | 加速比 ≥ 1.5x | 查询侧显存缓冲与检索参数对象复用（不动 itopk，避免伤 Recall） |
| 硬门槛 4 分 | 全过 | 只动一个质量旋钮，其余质量参数保持基线值 |
