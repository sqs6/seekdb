/* 仅用于本地语法自检的 cuda_runtime 桩头（hipVS 里由 cuvs/cuda_runtime.h 提供）。 */
#ifndef STUB_CUVS_CUDA_RUNTIME_H
#define STUB_CUVS_CUDA_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { cudaSuccess = 0, cudaErrorUnknown = 999 } cudaError_t;

typedef enum { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1,
               cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3,
               cudaMemcpyDefault = 4 } cudaMemcpyKind;

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaDeviceSynchronize(void);

#ifdef __cplusplus
}
#endif
#endif
