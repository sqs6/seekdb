/* 桩实现对外暴露的观测点（供 test_bridge.c 与 driver.c 共用）。 */
#ifndef STUB_OBSERVER_H
#define STUB_OBSERVER_H

#include <stddef.h>

/* 桩在 cuvsCagraBuild 里记录下来的建图参数 */
struct stub_index {
  long n;
  long dim;
  size_t graph_degree;
  size_t intermediate;
  size_t niter;
};

/* 最近一次建图产生的索引（由桩设置） */
extern struct stub_index *g_last_built;
/* 最近一次检索用到的检索参数（由桩记录） */
extern size_t g_last_itopk;
extern size_t g_last_width;

#endif /* STUB_OBSERVER_H */
