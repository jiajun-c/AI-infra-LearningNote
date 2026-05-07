#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void demo_basic_vmm(int device) {
    printf("basic vmm\n");
    size_t granularity = 0;
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device;

    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    printf("分配粒度: %zu bytes (%zu MB)\n", granularity,
           granularity / (1024 * 1024));
    
    CUmemAccessDesc assesDesc_{
        .location = {
            .type = CU_MEM_LOCATION_TYPE_DEVICE,
            .id   = 0,
        },
        .flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    };
}