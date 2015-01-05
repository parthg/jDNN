extern "C"

__global__ void kMul(double* a, double* b, double* dest, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n) {
    dest[idx] = a[idx] * b[idx];
  }
}
