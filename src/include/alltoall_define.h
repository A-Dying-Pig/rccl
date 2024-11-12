#pragma once

#include <stdio.h>

#ifdef NOLOG
#define LOG(...)
#else
#define LOG(fmt, ...)                   \
  {                                           \
    printf("LOG: " fmt "\n", ##__VA_ARGS__); \
    fflush(stdout);                       \
  }
#endif

#define MAX(x, y) ((x > y) ? x : y)
#define MIN(x, y) ((x < y) ? x : y)

#define SCALING_FACTOR_MIN 1
#define SCALING_FACTOR_MAX 1000000
#define SCALING_FACTOR_STEP 2

#define MAX_SUM_LIMIT 100
#define BENCHMARK_DIR "benchmark/"


typedef struct balance_data_t* BalancePtr;
typedef uint TransferMatrixElement;
typedef struct recv_data_t * RestorePtr;
typedef struct recv_data_t * DirectCpyPtr;
typedef uint * ChannelPtr;

#define MAX_BUFFER_SIZE_PER_RANK 2
#define MAX_ELEMENT_NUM 3

#define GPU_NUM_PER_SERVER 8
#define MAX_SERVER_NUM 16