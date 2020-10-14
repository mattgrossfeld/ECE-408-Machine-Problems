#include <wb.h>
#include <math.h> //For doing squares

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_RADIUS 1 //Given in lab description
#define MASK_WIDTH 3 //Given in lab description. Kernel size = 3x3x3. 
#define TILE_WIDTH ((2*MASK_RADIUS) + MASK_WIDTH) //Used for N_ds
//@@ Define constant memory for device kernel here
__constant__ float mask[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH]; //3D mask (hence width^3)

//Convolution code
__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  //Threads
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  
  //Blocks
  int bx = MASK_WIDTH * blockIdx.x;
  int by = MASK_WIDTH * blockIdx.y;
  int bz = MASK_WIDTH * blockIdx.z;
  
  //Output(?)
  int x_o = bx + tx;
  int y_o = by + ty;
  int z_o = bz + tz;
  
  //tx = dimension 1. ty for dimension 2. tz for dimension 3. Since its width^3, we start each dimension after the width.
  int kernelThread = tx + (MASK_WIDTH * ty) + ((int)pow(MASK_WIDTH,2) * tz); //Create a thread to each spot in the mask.
  
  __shared__ float N_ds[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH]; //3D Tiling   

  if (kernelThread < ((int)pow(TILE_WIDTH,2))) //Make sure our threads are within the tile bounds
  {
    int M = kernelThread % TILE_WIDTH; //M will be for x-axis tile
    int N = kernelThread / TILE_WIDTH; //N will be for y-axis tile
    N = N % TILE_WIDTH; //Need to make it loop.
    
    int x_i = bx + M;
    int y_i = by + N;
    int z_i = bz;
    
    x_i--; //Size - 1
    y_i--; //Size - 1
    
    //Load tile
    int i = 0;
    while (i < TILE_WIDTH)
    {
      //Increment our z-position every loop
      z_i = (bz-1) + i;
      //Bound check for input
      if ( (x_i >= 0) && (x_i < x_size) &&
       (y_i >= 0) && (y_i < y_size) &&
       (z_i >= 0) && (z_i < z_size)) {
        N_ds[M][N][i] = input[x_i + (y_i * x_size) + (z_i * y_size * x_size)]; //Load from input
      }
      else
      {
        N_ds[M][N][i] = 0; //Pad boundary with 0
      }
      i++;
    }
  }
    
    __syncthreads(); //Need all threads to catch up before progressing
  
      //Output bounds
  if ( (x_o >= 0) && (x_o < x_size) &&
       (y_o >= 0) && (y_o < y_size) &&
       (z_o >= 0) && (z_o < z_size)) {
    float res = 0;
    for (int x = 0; x < MASK_WIDTH; x++)
    {
      for (int y = 0; y < MASK_WIDTH; y++)
      {
        for (int z = 0; z < MASK_WIDTH; z++)
          res += (N_ds[x+tx][y+ty][z+tz])*mask[x+(y*MASK_WIDTH)+(z*(int)pow(MASK_WIDTH,2))];
      }
    }
    output[x_o + (y_o*x_size) + (z_o*x_size*y_size)] = res;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int mallocSize = x_size * y_size * z_size * sizeof(float);
  cudaMalloc((void**)&deviceInput, mallocSize);
  cudaMalloc((void**) &deviceOutput, mallocSize);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  int kernelSize = kernelLength * sizeof(float);
  cudaMemcpyToSymbol(mask, hostKernel, kernelSize, 0, cudaMemcpyHostToDevice); 
  //Above function loads constant values to host. This reduces memory loads by as much as 50% (from notes)
  cudaMemcpy(deviceInput, hostInput+3, mallocSize, cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  //We divide by MASK_WIDTH in all 3 dimensions (3D conv) for tiling purposes
  dim3 dimGrid(ceil(x_size/float(MASK_WIDTH)), ceil(y_size/float(MASK_WIDTH)), ceil(z_size/float(MASK_WIDTH)));  
  dim3 dimBlock(MASK_WIDTH, MASK_WIDTH, MASK_WIDTH);
  
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, mallocSize, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
