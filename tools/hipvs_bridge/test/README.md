# bridge 自测（无 GPU）

`seekdb_cuvs_bridge.c` 是 AMD 赛题链路里唯一直接调用 hipVS/cuVS 的地方。
在没有 AMD GPU 的机器上（开发机、CI、评审环境），可以用这里的 **cuVS 桩实现**
把它编译并跑起来，验证控制流与数据搬运是否正确。

```bash
bash tools/hipvs_bridge/test/run_test.sh
```

只依赖 `gcc`（或 clang）+ `libpthread`，**不需要 GPU、不需要 ROCm、不需要 hipVS**。

## 覆盖的用例

| 用例 | 验证点 |
| --- | --- |
| build | 句柄创建成功 |
| 单条检索 | 返回的 ID/距离序列正确；`itopk` 取基线同值 512；`search_width = 4` |
| 连续 5 次检索 | 常驻显存缓冲复用后数据不串（这是本次优化引入的新代码路径） |
| 批量检索 nq=64 | `topk=100` 时 `itopk ≥ 4*topk`；返回结果逐元素正确 |
| 序列化 / 反序列化 | 往返后仍可检索且结果一致；维度不匹配时**必须失败**（不能返回脏句柄） |
| 释放 | 不崩溃 |
| 环境变量 | `SEEKDB_CUVS_ITOPK_FLOOR` 生效 |

## 桩头的边界

`stub/` 下是**自测专用的假头文件**，签名按 cuVS C API 复刻，但语义是简化版：

- `cuvsCagraBuild` 把索引就地写进已建好的句柄（真实 cuVS 也是这个语义）；
- `cuvsCagraSearch` 不做真实图检索，直接按固定规则填 ID 与距离；
- `cuvsRMMAlloc` 返回普通内存，`cudaMemcpy` 就是 `memcpy`。

所以这套自测**验的是 bridge 自己的逻辑**（参数怎么算、缓冲怎么复用、数据怎么搬、
句柄怎么管），**不验** CAGRA 算法的正确性与性能 —— 那部分只能在真实 AMD GPU 上测，
用 `tools/hipvs_bridge/bench_cagra.sh`。
