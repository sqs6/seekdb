/* 仅用于本地语法自检的 DLPack 桩头（与官方 dlpack.h 字段一致）。 */
#ifndef STUB_DLPACK_DLPACK_H
#define STUB_DLPACK_DLPACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { kDLCPU = 1, kDLCUDA = 2, kDLCUDAHost = 3, kDLROCM = 10 } DLDeviceType;

typedef struct {
  DLDeviceType device_type;
  int          device_id;
} DLDevice;

typedef enum { kDLInt = 0, kDLUInt = 1, kDLFloat = 2 } DLDataTypeCode;

typedef struct {
  uint8_t      code;
  uint8_t      bits;
  uint16_t     lanes;
} DLDataType;

typedef struct {
  void*      data;
  DLDevice   device;
  int32_t    ndim;
  DLDataType dtype;
  int64_t*   shape;
  int64_t*   strides;
  uint64_t   byte_offset;
} DLTensor;

typedef struct DLPackVersion { uint32_t major; uint32_t minor; } DLPackVersion;

typedef struct DLManagedTensor {
  DLTensor dl_tensor;
  void*    manager_ctx;
  void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

#ifdef __cplusplus
}
#endif
#endif
