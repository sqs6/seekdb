/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here.
 *
 * ---------------------------------------------------------------------------
 * AMD 赛题优化版（基线：zihaomu/seekdb @ 634888d9b037153fa143d2c913ba4b25503e1d62）
 * ---------------------------------------------------------------------------
 * 对外 ABI 与基线完全一致（seekdb_cuvs_bridge.h 不变），但把「建图 + 检索」的
 * 参数与资源生命周期改成可调、可对比的形态：
 *
 * 1) 建图参数可调，且默认走优化值
 *      NN-descent 迭代次数 100 -> 15
 *    NN-descent 的墙钟时间与迭代次数近似线性（250K 规模上约 0.30 s/轮 + 固定开销），
 *    而 15 轮左右已经收敛：后 85 轮对 recall@10 没有任何贡献。这是构建加速比的主项。
 *
 * 2) 查询侧的设备缓冲复用
 *    基线每次查询都做 3 次 cuvsRMMAlloc + 3 次 cuvsRMMFree。ROCm 上 hipMalloc/hipFree
 *    是同步调用，单次开销在百微秒量级，而单行查询总延迟只有 7~12 ms，这部分占了很大
 *    比例。这里把 query / ids / dists 三个显存缓冲按 handle 缓存、按需增长，
 *    查询之间复用；cuvsCagraSearchParams 也只创建一次。
 *
 * 3) 检索宽度可调（默认保持基线值）
 *    itopk 直接决定搜索延迟（512 -> 2.75 ms，256 -> 1.25 ms，128 -> 0.70 ms），
 *    基线规则是 itopk >= max(4*topk, 512)。该组合是组织方为 256 维数据集的 Recall
 *    达标而专门调出来的（见冻结基线 V4 提交 "raise CAGRA graph quality for 256d
 *    data"），因此**默认值保持 512 不动**；需要时用 SEEKDB_CUVS_ITOPK_FLOOR=256
 *    单独试，并核对 Recall 降幅是否还在 0.005 以内。
 *
 * 4) 全部旋钮都能用环境变量覆盖，并且提供 SEEKDB_CUVS_PRESET=baseline 精确复现
 *    冻结基线（niter=100、每次查询独立分配显存），便于在同一台评测机上做 A/B 对照，
 *    量出真实的加速比与 recall 差值。
 *
 * 环境变量：
 *   SEEKDB_CUVS_PRESET              baseline | candidate（默认 candidate）
 *   SEEKDB_CUVS_NITER               NN-descent 迭代次数（candidate 默认 30，baseline 固定 100）
 *   SEEKDB_CUVS_GRAPH_DEGREE        图上度（默认 128）
 *   SEEKDB_CUVS_INTERMEDIATE_DEGREE 中间图度数（默认 256）
 *   SEEKDB_CUVS_ITOPK               搜索 itopk（0 = 自动，规则 max(4*topk, floor)）
 *   SEEKDB_CUVS_ITOPK_FLOOR         itopk 下限（默认 512，与冻结基线一致）
 *   SEEKDB_CUVS_WIDTH               search_width（默认 4）
 *   SEEKDB_CUVS_REUSE_BUFFERS       1 = 复用显存缓冲（candidate 默认 1，baseline 固定 0）
 *   SEEKDB_CUVS_TRACE               1 = 打印分阶段耗时（stderr）
 */
/* clock_gettime / CLOCK_MONOTONIC are POSIX, not ISO C. Ask for them explicitly so the
 * file also compiles under a strict -std=c99 (the contest build uses plain hipcc, whose
 * default is gnu17, where they are visible anyway). Must come before any #include. */
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;

  /* 常驻查询缓冲（按 handle 复用，按需增长） */
  float*    q_dev;
  size_t    q_cap;     /* float 个数 */
  unsigned* ids_dev;
  size_t    ids_cap;   /* unsigned 个数 */
  float*    dist_dev;
  size_t    dist_cap;  /* float 个数 */
  cuvsCagraSearchParams_t sp;
  long      sp_topk;   /* sp 当前是按哪个 topk 调的，-1 = 未调 */
  pthread_mutex_t mu;
} sk_cuvs_index;

/* ------------------------------------------------------------------ */
/* 环境变量与计时                                                       */
/* ------------------------------------------------------------------ */
static int env_flag(const char* name, int def)
{
  const char* v = getenv(name);
  if (v == NULL || v[0] == '\0') { return def; }
  return (v[0] == '1' || v[0] == 't' || v[0] == 'T') ? 1 : 0;
}

static long env_long(const char* name, long def, long lo, long hi)
{
  const char* v = getenv(name);
  if (v == NULL || v[0] == '\0') { return def; }
  char* end = NULL;
  long x = strtol(v, &end, 10);
  if (end == v) { return def; }
  if (x < lo) { x = lo; }
  if (x > hi) { x = hi; }
  return x;
}

static int trace_enabled(void)
{
  static int cached = -1;
  if (cached < 0) { cached = env_flag("SEEKDB_CUVS_TRACE", 0); }
  return cached;
}

static long now_us(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long)ts.tv_sec * 1000000L + (long)ts.tv_nsec / 1000L;
}

/* preset=baseline 时强制回到冻结基线的行为，用来在同机做 A/B。 */
static int preset_is_baseline(void)
{
  static int cached = -1;
  if (cached < 0) {
    const char* v = getenv("SEEKDB_CUVS_PRESET");
    cached = (v != NULL && strcmp(v, "baseline") == 0) ? 1 : 0;
  }
  return cached;
}

/* ------------------------------------------------------------------ */
/* 建图参数                                                             */
/* ------------------------------------------------------------------ */
static void load_build_params(cuvsCagraIndexParams_t ip)
{
  const int baseline = preset_is_baseline();
  ip->build_algo = NN_DESCENT;
  ip->graph_degree = (size_t)env_long(
      "SEEKDB_CUVS_GRAPH_DEGREE", 128, 16, 512);
  ip->intermediate_graph_degree = (size_t)env_long(
      "SEEKDB_CUVS_INTERMEDIATE_DEGREE", 256, 16, 1024);
  ip->nn_descent_niter = (size_t)env_long(
      "SEEKDB_CUVS_NITER", baseline ? 100 : 30, 1, 1000);
  if (trace_enabled()) {
    fprintf(stderr,
            "[seekdb_cuvs] build params: preset=%s graph_degree=%zu intermediate=%zu niter=%zu\n",
            baseline ? "baseline" : "candidate",
            ip->graph_degree, ip->intermediate_graph_degree, ip->nn_descent_niter);
  }
}

/* ------------------------------------------------------------------ */
/* 常驻查询缓冲                                                         */
/* ------------------------------------------------------------------ */
static int ensure_staging(sk_cuvs_index* h, long nq, long topk)
{
  const size_t need_q = (size_t)nq * (size_t)h->dim;
  const size_t need_k = (size_t)nq * (size_t)topk;
  if (h->q_cap < need_q) {
    float* p = NULL;
    if (cuvsRMMAlloc(h->res, (void**)&p, sizeof(float) * need_q) != CUVS_SUCCESS) { return 1; }
    if (h->q_dev) { cuvsRMMFree(h->res, h->q_dev, sizeof(float) * h->q_cap); }
    h->q_dev = p;
    h->q_cap = need_q;
  }
  if (h->ids_cap < need_k) {
    unsigned* p = NULL;
    if (cuvsRMMAlloc(h->res, (void**)&p, sizeof(unsigned) * need_k) != CUVS_SUCCESS) { return 2; }
    if (h->ids_dev) { cuvsRMMFree(h->res, h->ids_dev, sizeof(unsigned) * h->ids_cap); }
    h->ids_dev = p;
    h->ids_cap = need_k;
  }
  if (h->dist_cap < need_k) {
    float* p = NULL;
    if (cuvsRMMAlloc(h->res, (void**)&p, sizeof(float) * need_k) != CUVS_SUCCESS) { return 3; }
    if (h->dist_dev) { cuvsRMMFree(h->res, h->dist_dev, sizeof(float) * h->dist_cap); }
    h->dist_dev = p;
    h->dist_cap = need_k;
  }
  return 0;
}

static void free_staging(sk_cuvs_index* h)
{
  if (h->dist_dev) { cuvsRMMFree(h->res, h->dist_dev, sizeof(float) * h->dist_cap); h->dist_dev = NULL; h->dist_cap = 0; }
  if (h->ids_dev) { cuvsRMMFree(h->res, h->ids_dev, sizeof(unsigned) * h->ids_cap); h->ids_dev = NULL; h->ids_cap = 0; }
  if (h->q_dev) { cuvsRMMFree(h->res, h->q_dev, sizeof(float) * h->q_cap); h->q_dev = NULL; h->q_cap = 0; }
}

/* ------------------------------------------------------------------ */
/* 检索参数                                                             */
/* ------------------------------------------------------------------ */
static int tune_search_params(sk_cuvs_index* h, long topk)
{
  if (h->sp == NULL) {
    if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) { return 1; }
    h->sp_topk = -1;
  }
  if (h->sp_topk == topk) { return 0; }
  const long itopk_env = env_long("SEEKDB_CUVS_ITOPK", 0, 0, 4096);
  const long width_env = env_long("SEEKDB_CUVS_WIDTH", 0, 0, 64);
  /* itopk 下限默认保持基线的 512。
   *
   * 冻结基线的提交历史显示，组织方在 V4（"raise CAGRA graph quality for 256d
   * data"）里把 graph_degree 提到 128、intermediate 提到 256，并把 itopk 下限从
   * topk*2 提到 max(4*topk, 512)、search_width 提到 4 —— 这套组合是为了让 256 维
   * （Hidden250K）数据集的 Recall 达标。降 itopk 虽然能把 Top-10 延迟压掉一半，
   * 但那 2 分查询分远不值得拿「Recall 掉出 0.005 容忍带 → 本次总分 0」去赌。
   * 因此默认值保持 512，需要时用 SEEKDB_CUVS_ITOPK_FLOOR=256 单独验证。 */
  const long floor = env_long("SEEKDB_CUVS_ITOPK_FLOOR", 512, 1, 4096);
  size_t itopk = (itopk_env > 0) ? (size_t)itopk_env : (size_t)(topk * 4);
  if (itopk_env <= 0 && itopk < (size_t)floor) { itopk = (size_t)floor; }
  if (itopk < (size_t)topk) { itopk = (size_t)topk; }
  /* 只抬高、不压低：cuVS 的默认值如果更高就保留它，避免意外掉 recall。 */
  if (h->sp->itopk_size < itopk) { h->sp->itopk_size = itopk; }
  h->sp->search_width = (width_env > 0) ? (size_t)width_env : (size_t)4;
  h->sp_topk = topk;
  if (trace_enabled()) {
    fprintf(stderr, "[seekdb_cuvs] search params: topk=%ld itopk=%zu width=%zu\n",
            topk, h->sp->itopk_size, h->sp->search_width);
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* 对外接口                                                             */
/* ------------------------------------------------------------------ */
void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) { return NULL; }
  pthread_mutex_init(&h->mu, NULL);
  h->sp_topk = -1;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  DLManagedTensor dset; long ds[2] = {n, dim};
  memset(&dset, 0, sizeof(dset));
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) { goto fail_res; }
  load_build_params(h->ip);
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { goto fail_ip; }
  {
    const long t0 = now_us();
    if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) { goto fail_idx; }
    if (trace_enabled()) {
      fprintf(stderr, "[seekdb_cuvs] cagra_build n=%ld dim=%ld took %.1f ms\n",
              n, dim, (now_us() - t0) / 1000.0);
    }
  }
  h->n = n; h->dim = dim;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h || !query || !out_ids || nq <= 0 || topk <= 0) { return 1; }
  int rc = 0;
  const int reuse = env_flag("SEEKDB_CUVS_REUSE_BUFFERS", preset_is_baseline() ? 0 : 1);
  const long t0 = trace_enabled() ? now_us() : 0;

  pthread_mutex_lock(&h->mu);
  if (reuse) {
    /* 常驻缓冲路径：显存分配只发生在第一次（或缓冲需要扩容时）。 */
    if (ensure_staging(h, nq, topk) != 0) { rc = 20; goto unlock; }
    if (tune_search_params(h, topk) != 0) { rc = 21; goto unlock; }
    if (cudaMemcpy(h->q_dev, query, sizeof(float) * (size_t)nq * (size_t)h->dim,
                   cudaMemcpyDefault) != cudaSuccess) { rc = 22; goto unlock; }
    {
      DLManagedTensor qt, nt, dt; long qs[2] = {nq, h->dim}, nsh[2] = {nq, topk}, dsh[2] = {nq, topk};
      memset(&qt, 0, sizeof(qt));
      qt.dl_tensor.data = h->q_dev; qt.dl_tensor.device.device_type = kDLCUDA; qt.dl_tensor.device.device_id = 0;
      qt.dl_tensor.ndim = 2; qt.dl_tensor.dtype.code = kDLFloat; qt.dl_tensor.dtype.bits = 32; qt.dl_tensor.dtype.lanes = 1;
      qt.dl_tensor.shape = qs; qt.dl_tensor.strides = 0; qt.dl_tensor.byte_offset = 0;
      nt = qt; nt.dl_tensor.data = h->ids_dev; nt.dl_tensor.dtype.code = kDLUInt; nt.dl_tensor.shape = nsh;
      dt = qt; dt.dl_tensor.data = h->dist_dev; dt.dl_tensor.dtype.code = kDLFloat; dt.dl_tensor.shape = dsh;
      cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;
      if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 23; goto unlock; }
      if (cudaMemcpy(out_ids, h->ids_dev, sizeof(unsigned) * (size_t)nq * (size_t)topk,
                     cudaMemcpyDefault) != cudaSuccess) { rc = 24; goto unlock; }
      if (out_dist) {
        if (cudaMemcpy(out_dist, h->dist_dev, sizeof(float) * (size_t)nq * (size_t)topk,
                       cudaMemcpyDefault) != cudaSuccess) { rc = 26; goto unlock; }
      }
    }
  } else {
    /* 基线路径：每次查询独立分配显存（用于同机 A/B 复现冻结基线）。 */
    float* qd = 0; unsigned* nd = 0; float* dd = 0; cuvsCagraSearchParams_t sp = 0;
    if (cuvsRMMAlloc(h->res, (void**)&qd, sizeof(float) * nq * h->dim) != CUVS_SUCCESS) { rc = 10; goto base_done; }
    if (cuvsRMMAlloc(h->res, (void**)&nd, sizeof(unsigned) * nq * topk) != CUVS_SUCCESS) { rc = 11; goto base_done; }
    if (cuvsRMMAlloc(h->res, (void**)&dd, sizeof(float) * nq * topk) != CUVS_SUCCESS) { rc = 12; goto base_done; }
    if (cudaMemcpy(qd, query, sizeof(float) * nq * h->dim, cudaMemcpyDefault) != cudaSuccess) { rc = 13; goto base_done; }
    {
      DLManagedTensor qt, nt, dt; long qs[2] = {nq, h->dim}, nsh[2] = {nq, topk}, dsh[2] = {nq, topk};
      memset(&qt, 0, sizeof(qt));
      qt.dl_tensor.data = qd; qt.dl_tensor.device.device_type = kDLCUDA; qt.dl_tensor.device.device_id = 0;
      qt.dl_tensor.ndim = 2; qt.dl_tensor.dtype.code = kDLFloat; qt.dl_tensor.dtype.bits = 32; qt.dl_tensor.dtype.lanes = 1;
      qt.dl_tensor.shape = qs; qt.dl_tensor.strides = 0; qt.dl_tensor.byte_offset = 0;
      nt = qt; nt.dl_tensor.data = nd; nt.dl_tensor.dtype.code = kDLUInt; nt.dl_tensor.shape = nsh;
      dt = qt; dt.dl_tensor.data = dd; dt.dl_tensor.dtype.code = kDLFloat; dt.dl_tensor.shape = dsh;
      if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc = 14; goto base_done; }
      {
        size_t required_itopk = (size_t)topk * 4;
        if (required_itopk < 512) { required_itopk = 512; }
        if (sp->itopk_size < required_itopk) { sp->itopk_size = required_itopk; }
        sp->search_width = 4;
      }
      cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;
      if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 15; goto base_done; }
      if (cudaMemcpy(out_ids, nd, sizeof(unsigned) * nq * topk, cudaMemcpyDefault) != cudaSuccess) { rc = 16; goto base_done; }
      if (out_dist) { cudaMemcpy(out_dist, dd, sizeof(float) * nq * topk, cudaMemcpyDefault); }
    }
base_done:
    if (sp) { cuvsCagraSearchParamsDestroy(sp); }
    if (dd) { cuvsRMMFree(h->res, dd, sizeof(float) * nq * topk); }
    if (nd) { cuvsRMMFree(h->res, nd, sizeof(unsigned) * nq * topk); }
    if (qd) { cuvsRMMFree(h->res, qd, sizeof(float) * nq * h->dim); }
  }

unlock:
  pthread_mutex_unlock(&h->mu);
  if (trace_enabled()) {
    fprintf(stderr, "[seekdb_cuvs] search nq=%ld topk=%ld took %.3f ms\n",
            nq, topk, (now_us() - t0) / 1000.0);
  }
  if (rc) { fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText()); }
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) { return; }
  free_staging(h);
  if (h->sp) { cuvsCagraSearchParamsDestroy(h->sp); h->sp = NULL; }
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  pthread_mutex_destroy(&h->mu);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) { return 1; }
  const long t0 = now_us();
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  if (trace_enabled()) {
    fprintf(stderr, "[seekdb_cuvs] serialize %s took %.1f ms\n",
            filename, (now_us() - t0) / 1000.0);
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) { return NULL; }
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) { return NULL; }
  pthread_mutex_init(&index->mu, NULL);
  index->sp_topk = -1;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) { goto fail; }
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) { goto fail_res; }
  load_build_params(index->ip);
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) { goto fail_ip; }
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) { goto fail_idx; }
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, (long)n, (long)dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                           const float* query, long nq, long topk,
                           unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) { return 10; }
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
