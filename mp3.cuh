#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 4 //Tile width is a #define constant. Originally 8, but 4 seems to be the only value that works

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float M[TILE_WIDTH][TILE_WIDTH]; //Subtile M
  __shared__ float N[TILE_WIDTH][TILE_WIDTH]; //Subtile N
  
  int xBlock = blockIdx.x; int yBlock = blockIdx.y; //Block indexes
  int xThread = threadIdx.x; int yThread = threadIdx.y; //Thread indexes
  
  int row = yBlock * TILE_WIDTH + yThread; //Calculate row as done in previous MPs
  int col = xBlock * TILE_WIDTH + xThread; //Calculate columns as done in previous MPs
  
  if ((row < numCRows) && (col < numCColumns))
  {
    float pValue = 0; //Our calculated value is default 0.
  
    for (int i = 0; i < numAColumns/TILE_WIDTH; ++i) //Need to loop through M and N tiles. Need to use numAColumns for same reason as MP2
    {
      M[yThread][xThread] = A[row*numAColumns + i*TILE_WIDTH + xThread];
      N[yThread][xThread] = B[(i*TILE_WIDTH+yThread)*numBColumns+col];
      __syncthreads(); //need everything to be sync'd.
      for (int j = 0; j < TILE_WIDTH; ++j) //Loop t hrough the tile width
      {
        pValue += M[yThread][j] * N[j][xThread]; //Need to calculate new value
      }
      __syncthreads(); //Need to sync here
    }
    C[row*numCColumns+col] = pValue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numARows * numBColumns * sizeof(float)); //Regular malloc because its host NOT device.
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int aSize = numARows * numAColumns * sizeof(float);
  int bSize = numBRows * numBColumns * sizeof(float);
  int cSize = numCRows * numCColumns * sizeof(float);
  cudaMalloc((void**)&deviceA, aSize);
  cudaMalloc((void**)&deviceB, bSize);
  cudaMalloc((void**)&deviceC, cSize);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, aSize, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, bSize, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numCColumns/float(TILE_WIDTH)), ceil(numCRows/float(TILE_WIDTH)), 1); 
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC, 
                                        numARows, numAColumns,
                                        numBRows, numBColumns,
                                        numCRows, numCColumns
                                       );
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, cSize, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree((void*)deviceA);
  cudaFree((void*)deviceB);
  cudaFree((void*)deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
