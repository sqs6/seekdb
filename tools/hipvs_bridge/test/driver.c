/* 打印 stub 观测到的建图/检索参数 —— 用来对比「冻结基线的 bridge」与
 * 「本实现的 baseline 预设」是否**逐项一致**。
 * 整个 A/B 方法论（用 baseline 预设当对照组）都建立在这个等式上。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "seekdb_cuvs_bridge.h"
#include "stub_observer.h"

/* 桩暴露的观测点声明见 stub_observer.h */

int main(void) {
  const long n = 1000, dim = 8;
  float* base = (float*)malloc(sizeof(float) * n * dim);
  for (long i = 0; i < n * dim; ++i) { base[i] = (float)(i % 37) * 0.25f; }

  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) { printf("build failed\n"); return 1; }

  float q[8];
  for (int i = 0; i < 8; ++i) { q[i] = (float)i; }
  unsigned ids[10];
  float dist[10];
  if (seekdb_cuvs_search(h, q, 1, 10, ids, dist) != 0) {
    printf("search failed\n");
    return 1;
  }

  printf("build: niter=%zu graph_degree=%zu intermediate=%zu\n",
         g_last_built->niter, g_last_built->graph_degree, g_last_built->intermediate);
  printf("search: itopk=%zu width=%zu\n", g_last_itopk, g_last_width);

  /* 检索结果的形状也要一致（桩实现是确定性的） */
  unsigned long sum = 0;
  for (int i = 0; i < 10; ++i) { sum += ids[i]; }
  printf("ids_sum=%lu dist0=%.1f dist9=%.1f\n", sum, dist[0], dist[9]);

  seekdb_cuvs_free(h);
  free(base);
  return 0;
}
