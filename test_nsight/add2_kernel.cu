#include <stdio.h>
#include <iostream>
__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n) {
    //printf("i start: %d \n ",blockIdx.x * blockDim.x + threadIdx.x);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
        //printf("idx: %d , val a: %f: \n ",i,a[i]);		
    }
}

void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);

    cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start, 0); // Record an event in the default 


    add2_kernel<<<grid, block>>>(c, a, b, n);

cudaEventRecord(stop, 0); // Record another event after the kernel launch
cudaEventSynchronize(stop); // Wait for the event to be recorded
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
std::cout << "Elapsed time: " << elapsedTime << " ms." << std::endl;
cudaEventDestroy(start);
cudaEventDestroy(stop);

}

// int main(void){
	
// 	float a[3]={1,2,3};
// 	float b[3]={4,5,6};
// 	float c[3]={0,0,0};
// 	launch_add2(a,b,c,3);


	
// return 0;

// }
