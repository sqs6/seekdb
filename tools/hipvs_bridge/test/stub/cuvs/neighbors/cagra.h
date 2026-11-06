/* 仅用于本地语法自检的 cuVS CAGRA 桩头。 */
#ifndef STUB_CUVS_NEIGHBORS_CAGRA_H
#define STUB_CUVS_NEIGHBORS_CAGRA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <cuvs/core/c_api.h>
#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { SINGLE_LEVEL = 0, MULTI_LEVEL = 1, IVF_PQ = 2, NN_DESCENT = 3 } cuvsCagraBuildAlgo;

typedef struct cuvsCagraIndexParams* cuvsCagraIndexParams_t;
typedef struct cuvsCagraIndex*      cuvsCagraIndex_t;
typedef struct cuvsCagraSearchParams* cuvsCagraSearchParams_t;

/* 真实的 cuVS C API 里 cuvsCagraIndex 是完整类型（定义在 cuvs 的 C++ 头里），
 * 这里补一个占位定义，桩实现才好按指针强转。 */
struct cuvsCagraIndex { int unused; };

struct cuvsCagraIndexParams {
  cuvsCagraBuildAlgo build_algo;
  size_t             graph_degree;
  size_t             intermediate_graph_degree;
  size_t             nn_descent_niter;
  uint64_t           seed;
};

struct cuvsCagraSearchParams {
  size_t itopk_size;
  size_t search_width;
  size_t min_iterations;
  size_t max_iterations;
};

int cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* params);
int cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t params);
int cuvsCagraIndexCreate(cuvsCagraIndex_t* index);
int cuvsCagraIndexDestroy(cuvsCagraIndex_t index);
int cuvsCagraBuild(cuvsResources_t res, cuvsCagraIndexParams_t params,
                   DLManagedTensor* dataset, cuvsCagraIndex_t index);
int cuvsCagraSearch(cuvsResources_t res, cuvsCagraSearchParams_t search_params,
                    cuvsCagraIndex_t index, DLManagedTensor* queries,
                    DLManagedTensor* neighbors, DLManagedTensor* distances,
                    cuvsFilter filter);
int cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* params);
int cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t params);
int cuvsCagraSerialize(cuvsResources_t res, const char* filename,
                       cuvsCagraIndex_t index, bool include_index);
int cuvsCagraDeserialize(cuvsResources_t res, const char* filename, cuvsCagraIndex_t index);
int cuvsCagraIndexGetSize(cuvsCagraIndex_t index, int64_t* size);
int cuvsCagraIndexGetDims(cuvsCagraIndex_t index, int64_t* dims);

#ifdef __cplusplus
}
#endif
#endif
