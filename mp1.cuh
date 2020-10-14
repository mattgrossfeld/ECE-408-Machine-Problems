// MP 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * (2*blockDim.x) + threadIdx.x;
  if (i < len)
    out[i] = in1[i] + in2[i];

  i+= blockDim.x; //Increment so we can do two at a time.
  if (i < len)
    out[i] = in1[i] + in2[i];
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  float *deviceInput1;
  float *deviceInput2;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  int mallocSize = inputLength * sizeof(float); //The size of each memory allocation
  cudaMalloc((void**)&deviceInput1, mallocSize); //Must have pointer to pointer.
  cudaMalloc((void**)&deviceInput2, mallocSize);
  cudaMalloc((void**)&deviceOutput, mallocSize);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, mallocSize, cudaMemcpyHostToDevice); //Need to copy the inputs
  cudaMemcpy(deviceInput2, hostInput2, mallocSize, cudaMemcpyHostToDevice); // From the host to device.
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(inputLength/512.0), 1, 1); //Grid dimensions. 512 blocks in each grid.
  dim3 DimBlock(512, 1, 1); //Block dimensions. 512 threads in each block.
  
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength); //Perform vec addition.
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, mallocSize, cudaMemcpyDeviceToHost); //Copy the output.
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree((void*)deviceInput1);
  cudaFree((void*)deviceInput2);
  cudaFree((void*)deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
