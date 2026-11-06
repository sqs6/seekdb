/* bridge 功能自测：在没有 AMD GPU 的机器上，用 cuVS 桩实现验证
 * tools/hipvs_bridge/seekdb_cuvs_bridge.c 的控制流与数据搬运是否正确。
 * 覆盖：建图参数、检索参数(itopk/width)、缓冲复用、批量检索、序列化/反序列化、释放。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "seekdb_cuvs_bridge.h"

/* 桩实现里暴露出来的观测点 */
extern size_t g_last_itopk;
extern size_t g_last_width;

static int g_fail = 0;
#define CHECK(cond, fmt, ...)                                                   \
  do {                                                                          \
    if (!(cond)) {                                                              \
      printf("  [FAIL] " fmt "\n", ##__VA_ARGS__);                              \
      g_fail++;                                                                 \
    } else {                                                                    \
      printf("  [ ok ] " fmt "\n", ##__VA_ARGS__);                              \
    }                                                                           \
  } while (0)

int main(void) {
  const long n = 1000, dim = 8;
  float* base = (float*)malloc(sizeof(float) * n * dim);
  for (long i = 0; i < n * dim; ++i) { base[i] = (float)(i % 97) * 0.5f; }

  printf("== 1. build ==\n");
  void* h = seekdb_cuvs_build(base, n, dim);
  CHECK(h != NULL, "build 返回有效句柄");
  if (!h) { free(base); return 1; }

  printf("== 2. 单条检索（nq=1）==\n");
  {
    unsigned ids[10];
    float dist[10];
    float q[8];
    for (int i = 0; i < 8; ++i) { q[i] = (float)i; }
    int rc = seekdb_cuvs_search(h, q, 1, 10, ids, dist);
    CHECK(rc == 0, "search rc=0（实际 %d）", rc);
    CHECK(ids[0] == 0 && ids[1] == 1 && ids[9] == 9, "返回的 id 序列符合预期");
    CHECK(dist[0] == 0.0f && dist[9] == 9.0f, "返回的距离符合预期");
    /* 基线规则：itopk >= max(4*topk, 512) → topk=10 时为 512 */
    CHECK(g_last_itopk == 512, "itopk = 512（基线同值，实际 %zu）", g_last_itopk);
    CHECK(g_last_width == 4, "search_width = 4（实际 %zu）", g_last_width);
  }

  printf("== 3. 连续多次检索（验证常驻缓冲复用不串数据）==\n");
  {
    int ok = 1;
    for (int t = 0; t < 5; ++t) {
      unsigned ids[10];
      float dist[10];
      float q[8];
      for (int i = 0; i < 8; ++i) { q[i] = (float)(t * 3 + i); }
      if (seekdb_cuvs_search(h, q, 1, 10, ids, dist) != 0) { ok = 0; break; }
      for (int k = 0; k < 10; ++k) { if (dist[k] != (float)k) { ok = 0; break; } }
    }
    CHECK(ok, "5 轮检索结果都正确（缓冲复用未污染数据）");
  }

  printf("== 4. 批量检索（nq=64）==\n");
  {
    unsigned ids[64 * 100];
    float dist[64 * 100];
    float* q = (float*)malloc(sizeof(float) * 64 * dim);
    for (long i = 0; i < 64 * dim; ++i) { q[i] = (float)(i % 31); }
    int rc = seekdb_cuvs_search(h, q, 64, 100, ids, dist);
    CHECK(rc == 0, "批量 search rc=0（实际 %d）", rc);
    int ok = 1;
    for (long row = 0; row < 64; ++row) {
      for (int k = 0; k < 100; ++k) { if (dist[row * 100 + k] != (float)k) { ok = 0; break; } }
    }
    CHECK(ok, "批量结果正确");
    CHECK(g_last_itopk >= 400, "批量时 itopk >= 4*topk=400（实际 %zu）", g_last_itopk);
    free(q);
  }

  printf("== 5. 序列化 / 反序列化 ==\n");
  {
    const char* path = "/tmp/stub_cagra_snapshot.bin";
    int rc = seekdb_cuvs_serialize(h, path);
    CHECK(rc == 0, "serialize rc=0（实际 %d）", rc);
    void* h2 = seekdb_cuvs_deserialize(path, n, dim);
    CHECK(h2 != NULL, "deserialize 返回有效句柄");
    if (h2) {
      unsigned ids[10];
      float dist[10];
      float q[8];
      for (int i = 0; i < 8; ++i) { q[i] = (float)i; }
      int rc2 = seekdb_cuvs_search(h2, q, 1, 10, ids, dist);
      CHECK(rc2 == 0, "反序列化后的句柄可检索");
      CHECK(ids[3] == 3, "反序列化后的结果正确");
      seekdb_cuvs_free(h2);
    }
    void* bad = seekdb_cuvs_deserialize(path, n, dim + 1);
    CHECK(bad == NULL, "维度不匹配时反序列化失败（不能返回脏句柄）");
  }

  printf("== 6. 释放 ==\n");
  {
    seekdb_cuvs_free(h);
    CHECK(1, "free 未崩溃");
  }

  printf("== 7. 参数覆盖（环境变量）==\n");
  {
    setenv("SEEKDB_CUVS_NITER", "42", 1);
    setenv("SEEKDB_CUVS_ITOPK_FLOOR", "256", 1);
    void* h3 = seekdb_cuvs_build(base, n, dim);
    CHECK(h3 != NULL, "带环境变量 build 成功");
    if (h3) {
      unsigned ids[10];
      float dist[10];
      float q[8];
      for (int i = 0; i < 8; ++i) { q[i] = (float)i; }
      seekdb_cuvs_search(h3, q, 1, 10, ids, dist);
      CHECK(g_last_itopk == 256, "itopk 下限被环境变量改成 256（实际 %zu）", g_last_itopk);
      seekdb_cuvs_free(h3);
    }
  }

  free(base);
  printf("\n%s（失败 %d 项）\n", g_fail == 0 ? "===== 全部通过 =====" : "===== 有失败 =====", g_fail);
  return g_fail == 0 ? 0 : 1;
}
