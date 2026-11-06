/* 仅用于本地语法自检的 cuVS C API 桩头（评测机上是真 hipVS 头文件）。
 * 签名按 cuVS/hipVS 的公开 C API 复刻，字段名与类型与真实头文件一致。 */
#ifndef STUB_CUVS_CORE_C_API_H
#define STUB_CUVS_CORE_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUVS_SUCCESS 0

typedef struct cuvsResources* cuvsResources_t;

typedef enum { NO_FILTER = 0, MULTI_RESULTS_FILTER = 1 } cuvsFilterType;

typedef struct {
  cuvsFilterType type;
  uintptr_t     addr;
} cuvsFilter;

const char* cuvsGetLastErrorText(void);

int cuvsResourcesCreate(cuvsResources_t* res);
int cuvsResourcesDestroy(cuvsResources_t res);

int cuvsRMMAlloc(cuvsResources_t res, void** addr, size_t bytes);
int cuvsRMMFree(cuvsResources_t res, void* addr, size_t bytes);

#ifdef __cplusplus
}
#endif
#endif
