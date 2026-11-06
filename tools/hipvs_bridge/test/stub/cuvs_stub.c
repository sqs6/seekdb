/* cuVS/CUDA API 的桩实现：只求「能跑通 bridge 的控制流与数据搬运」，
 * 不模拟任何真实的 CAGRA 算法。用于在没有 AMD GPU 的机器上验证
 * tools/hipvs_bridge/seekdb_cuvs_bridge.c 的逻辑正确性。 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#include "stub_observer.h"

#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif

static const char* g_last_error = "stub: no error";
/* 假的索引与观测点定义见 stub_observer.h */
struct stub_index* g_last_built = NULL;
size_t g_last_itopk = 0;
size_t g_last_width = 0;

const char* cuvsGetLastErrorText(void) { return g_last_error; }

int cuvsResourcesCreate(cuvsResources_t* res) {
  cuvsResources_t r = (cuvsResources_t)calloc(1, 64);
  if (!r) { return 1; }
  *res = r;
  return CUVS_SUCCESS;
}

int cuvsResourcesDestroy(cuvsResources_t res) { free(res); return CUVS_SUCCESS; }

int cuvsRMMAlloc(cuvsResources_t res, void** addr, size_t bytes) {
  (void)res;
  /* 桩实现返回普通内存：bridge 的数据搬运逻辑照样成立 */
  *addr = malloc(bytes ? bytes : 1);
  return *addr ? CUVS_SUCCESS : 1;
}

int cuvsRMMFree(cuvsResources_t res, void* addr, size_t bytes) {
  (void)res; (void)bytes;
  free(addr);
  return CUVS_SUCCESS;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
  (void)kind;
  memcpy(dst, src, count);
  return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void) { return cudaSuccess; }

int cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params) {
  cuvsCagraIndexParams_t p = (cuvsCagraIndexParams_t)calloc(1, sizeof(*p));
  if (!p) { return 1; }
  *params = p;
  return CUVS_SUCCESS;
}

int cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params) { free(params); return CUVS_SUCCESS; }

int cuvsCagraIndexCreate(cuvsCagraIndex_t* index) {
  cuvsCagraIndex_t i = (cuvsCagraIndex_t)calloc(1, sizeof(struct stub_index));
  if (!i) { return 1; }
  *index = i;
  return CUVS_SUCCESS;
}

int cuvsCagraIndexDestroy(cuvsCagraIndex_t index) { free(index); return CUVS_SUCCESS; }

int cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params) {
  cuvsCagraSearchParams_t p = (cuvsCagraSearchParams_t)calloc(1, sizeof(*p));
  if (!p) { return 1; }
  p->itopk_size = 32;   /* cuVS 的默认值 */
  p->search_width = 1;
  *params = p;
  return CUVS_SUCCESS;
}

int cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params) { free(params); return CUVS_SUCCESS; }

int cuvsCagraBuild(cuvsResources_t res, cuvsCagraIndexParams_t params,
                   DLManagedTensor* dataset, cuvsCagraIndex_t index) {
  (void)res;
  /* ??????????? cuvsCagraIndexCreate ???Build ????? */
  struct stub_index* s = (struct stub_index*)index;
  if (!s) { return 1; }
  s->n = dataset->dl_tensor.shape[0];
  s->dim = dataset->dl_tensor.shape[1];
  s->graph_degree = params->graph_degree;
  s->intermediate = params->intermediate_graph_degree;
  s->niter = params->nn_descent_niter;
  g_last_built = s;
  return CUVS_SUCCESS;
}

int cuvsCagraSearch(cuvsResources_t res, cuvsCagraSearchParams_t sp, cuvsCagraIndex_t index,
                    DLManagedTensor* queries, DLManagedTensor* neighbors,
                    DLManagedTensor* distances, cuvsFilter filter) {
  (void)res; (void)filter;
  struct stub_index* s = (struct stub_index*)index;
  const long nq = queries->dl_tensor.shape[0];
  const long topk = neighbors->dl_tensor.shape[1];
  unsigned* nid = (unsigned*)neighbors->dl_tensor.data;
  float* ndist = (float*)distances->dl_tensor.data;
  const long n = s ? s->n : 0;
  for (long q = 0; q < nq; ++q) {
    for (long k = 0; k < topk; ++k) {
      unsigned id = (unsigned)((q * topk + k) % (n > 0 ? n : 1));
      nid[q * topk + k] = id;
      ndist[q * topk + k] = (float)k;
    }
  }
  if (sp) {
    g_last_itopk = sp->itopk_size;
    g_last_width = sp->search_width;
  }
  return CUVS_SUCCESS;
}

int cuvsCagraSerialize(cuvsResources_t res, const char* filename, cuvsCagraIndex_t index, bool include_index) {
  (void)res; (void)include_index;
  struct stub_index* s = (struct stub_index*)index;
  FILE* f = fopen(filename, "wb");
  if (!f) { g_last_error = "stub: open failed"; return 1; }
  if (s) { fwrite(&s->n, sizeof(s->n), 1, f); fwrite(&s->dim, sizeof(s->dim), 1, f); }
  fclose(f);
  return CUVS_SUCCESS;
}

int cuvsCagraDeserialize(cuvsResources_t res, const char* filename, cuvsCagraIndex_t index) {
  (void)res;
  struct stub_index* s = (struct stub_index*)index;
  if (!s) { return 1; }
  FILE* f = fopen(filename, "rb");
  if (!f) { g_last_error = "stub: open failed"; return 1; }
  if (fread(&s->n, sizeof(s->n), 1, f) != 1) { /* stub 文件可能只有部分内容 */ }
  if (fread(&s->dim, sizeof(s->dim), 1, f) != 1) { /* 同上 */ }
  fclose(f);
  return CUVS_SUCCESS;
}

int cuvsCagraIndexGetSize(cuvsCagraIndex_t index, int64_t* size) {
  struct stub_index* s = (struct stub_index*)index;
  *size = s ? s->n : 0;
  return CUVS_SUCCESS;
}

int cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int64_t* dims) {
  struct stub_index* s = (struct stub_index*)index;
  *dims = s ? s->dim : 0;
  return CUVS_SUCCESS;
}
