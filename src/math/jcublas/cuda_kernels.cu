extern "C"
__global__ void kMul(double* a, double* b, double* dest, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n) {
    dest[idx] = a[idx] * b[idx];
  }
}


extern "C"
__global__ void kFillArray(double* a, int m, double* dest, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n) {
    dest[idx] = a[idx % m];
  }
}


extern "C"
__global__ void kFill(double v, double* dest, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n) {
    dest[idx] = v;
  }
}

extern "C"
__global__ void kSigmoid(double* a, double* dest, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n) {
    dest[idx] = 1/(1+ exp(-1*a[idx]));
  }
}

extern "C"
__global__ void kPow(double* a, double y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n) {
    a[idx] = pow(a[idx], y);
  }
}

extern "C"
__global__ void kInverseElements(double* a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n) {
    a[idx] = (a[idx]==0.0)?0.0:1.0/a[idx];
  }
}

extern "C"
__global__ void kSqrt(double* a, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n) {
    a[idx] = sqrt(a[idx]);
  }
}

extern "C"
__global__ void kDivByColumnVector(double *a, int m, double* dest, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n) {
    dest[idx] = (a[idx/m]==0.0)?0.0:dest[idx]/a[idx/m];
  }
}

extern "C"
__global__ void kMulByColumnVector(double *a, int m, double* dest, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n) {
    dest[idx] = dest[idx]*a[idx/m];
  }
}
