#include <stdio.h>

// void c_hello(){
//     printf("Hello World!\n");
// }


__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    c_hello();
    cuda_hello<<<1,1>>>();
    return 0;
}