#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"

#define BYTES_TO_KILO_BYTES(x) x / (1024)
#define BYTES_TO_MEGA_BYTES(x) BYTES_TO_KILO_BYTES(x) / (1024)
#define BYTES_TO_GIGA_BYTES(x) BYTES_TO_MEGA_BYTES(x) / (1024)

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif

using namespace std;

int main() {
    hipDeviceProp_t devProp;
    hipError_t hip_error;
    int hip_device_id=0;
    size_t free_memory, total_memory;

    HIP_ASSERT(hipSetDevice(hip_device_id));
    hip_error = hipGetDeviceProperties(&devProp, hip_device_id);
    hip_error = hipMemGetInfo(&free_memory, &total_memory);

    cout << " System major    : "       << devProp.major                                  << endl;
    cout << " System minor    : "       << devProp.minor                                  << endl;
    cout << " agent prop name : "       << devProp.name                                   << endl;
    cout << " asicRevision    : "       << devProp.asicRevision                           << endl;
    cout << "---------------------------------------------------------------------------" << endl;
    cout << " managedMemory        : "  << devProp.managedMemory                          << endl;
    cout << " pageableMemoryAccess : "  << devProp.pageableMemoryAccess                   << endl;
    cout << " canMapHostMemory     : "  << devProp.canMapHostMemory                       << endl;
    cout << " totalConstMem        : "  << BYTES_TO_MEGA_BYTES(devProp.totalConstMem)     << " MB" << endl;
    cout << " sharedMemPerBlock    : "  << BYTES_TO_KILO_BYTES(devProp.sharedMemPerBlock) << " KB" << endl;
    cout << " totalGlobalMem       : "  << BYTES_TO_MEGA_BYTES(devProp.totalGlobalMem)    << " MB" << endl;
    cout << " l2CacheSize          : "  << BYTES_TO_MEGA_BYTES(devProp.l2CacheSize)       << " MB" << endl;
    cout << "---------------------------------------------------------------------------" << endl;
    cout << " regsPerBlock         : "  << devProp.regsPerBlock                           << endl;
    cout << "---------------------------------------------------------------------------" << endl;
    cout << " maxThreadsPerBlock   : "  << devProp.maxThreadsPerBlock                     << endl;
    cout << " maxThreadsDim[0]     : "  << devProp.maxThreadsDim[0]                       << endl;
    cout << " maxThreadsDim[1]     : "  << devProp.maxThreadsDim[1]                       << endl;
    cout << " maxThreadsDim[2]     : "  << devProp.maxThreadsDim[2]                       << endl;
    cout << " maxGridSize[0]       : "  << devProp.maxGridSize[0]                         << endl;
    cout << " maxGridSize[1]       : "  << devProp.maxGridSize[1]                         << endl;
    cout << " maxGridSize[2]       : "  << devProp.maxGridSize[2]                         << endl;
    cout << "---------------------------------------------------------------------------" << endl;
    cout << " Total Memory         : "  << BYTES_TO_MEGA_BYTES(total_memory)              << endl;
    cout << " Free Memory          : "  << BYTES_TO_MEGA_BYTES(free_memory)               << endl;
    cout << "---------------------------------------------------------------------------" << endl;
    cout << " gcnArchName          : "  << devProp.gcnArchName                            << endl;
    cout << " computeMode          : "  << devProp.computeMode                            << endl;
    cout << "---------------------------------------------------------------------------" << endl;
    cout << " isMultiGpuBoard      : "  << devProp.isMultiGpuBoard                        << endl;
    cout << "---------------------------------------------------------------------------" << endl;
    cout << " pciDomainID          : "  << devProp.pciDomainID                            << endl;
    cout << " pciBusID             : "  << devProp.pciBusID                               << endl;
    cout << " pciDeviceID          : "  << devProp.pciDeviceID                            << endl;
    cout << "---------------------------------------------------------------------------" << endl;
    cout << "hip Device prop succeeded " << endl ;


    cout << "---------------------------------------------------------------------------" << endl;
    return 0;
}