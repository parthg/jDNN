package math.jcublas;

//import jcuda.*;
//import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.cublasStatus;

import static jcuda.driver.JCudaDriver.*;
import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import math.DMatrix;
import math.CUDAMatrix;

public class SimpleCuBlas {
  public static final int THREADS_PER_BLOCK = 1024;
  public static int cudaCount = 0;
  public static void close() {
    JCublas.cublasShutdown();
  }

  public static void reset() {
    JCublas.cublasShutdown();
    JCuda.cudaDeviceReset();
  }

  public static int getGridDim(int n) {
    return ((n-1)/THREADS_PER_BLOCK)+1;
  }
  public static int getBlockDim(int n) {
    return Math.min(n,THREADS_PER_BLOCK);
  }

  // cublasAlloc &  cublasSetVector H->D
  public static Pointer alloc(DMatrix m) {
//    System.out.printf("Allocationg memory\n");
    JCublas.cublasInit();
    Pointer ret = new Pointer();
    Pointer toData = Pointer.to(m.data()).withByteOffset(m.offset() * m.elemSize());
    int err = JCublas.cublasAlloc(
        m.length(),
        m.elemSize(),
        ret);
//    assert(err == cublasStatus.CUBLAS_STATUS_INVALID_VALUE):"There was a problem in cuBlasAlloc: Either matrix length <= 0 or element size (float/double) <= 0.\n";
//    assert(err == cublasStatus.CUBLAS_STATUS_SUCCESS):"There was a problem in cuBlasAlloc.\n";
    JCublas.cublasSetVector(
        m.length(),
        m.elemSize(),
        toData,
        1,
        ret,
        1);
    cudaCount++;
    return ret;
  }

  public static Pointer alloc(DMatrix m, int offset, int length) {
//    System.out.printf("Allocationg memory\n");
    JCublas.cublasInit();
    Pointer ret = new Pointer();
    Pointer toData = Pointer.to(m.data()).withByteOffset(offset * m.elemSize());
    int err = JCublas.cublasAlloc(
        length,
        m.elemSize(),
        ret);
//    assert(err == cublasStatus.CUBLAS_STATUS_INVALID_VALUE):"There was a problem in cuBlasAlloc: Either matrix length <= 0 or element size (float/double) <= 0.\n";
//    assert(err == cublasStatus.CUBLAS_STATUS_SUCCESS):"There was a problem in cuBlasAlloc.\n";
    JCublas.cublasSetVector(
        length,
        m.elemSize(),
        toData,
        1,
        ret,
        1);
    cudaCount++;
    return ret;
  }
  
  public static Pointer alloc(double[] arr) {
    JCublas.cublasInit();
    Pointer ret = new Pointer();
    Pointer toData = Pointer.to(arr).withByteOffset(0 * Sizeof.DOUBLE);
    int err = JCublas.cublasAlloc(
        arr.length,
        Sizeof.DOUBLE,
        ret);
//    assert(err == cublasStatus.CUBLAS_STATUS_INVALID_VALUE):"There was a problem in cuBlasAlloc: Either matrix length <= 0 or element size (float/double) <= 0.\n";
//    assert(err == cublasStatus.CUBLAS_STATUS_SUCCESS):"There was a problem in cuBlasAlloc.\n";
    JCublas.cublasSetVector(
        arr.length, // size of array
        Sizeof.DOUBLE, // size of int
        toData,
        1,
        ret,
        1);
    cudaCount++;
    return ret;
  }

  public static Pointer alloc(int v) {
    Pointer ptr = new Pointer();
    ptr = Pointer.to(new int[]{v});
    cudaCount++;
    return ptr;
  }
  
  public static void updateData(Pointer ptr, double[] arr) {
    JCublas.cublasInit();
    Pointer toData = Pointer.to(arr).withByteOffset(0 * Sizeof.DOUBLE);
    JCublas.cublasSetVector(
        arr.length, // size of array
        Sizeof.DOUBLE, // size of int
        toData,
        1,
        ptr,
        1);
  }

  // cublasGetVector D->H
  public static void getData(DMatrix arr,Pointer from,Pointer to) {
    JCublas.cublasInit();
    assert arr.length() == arr.data().length;
      JCublas.cublasGetVector(
          arr.length(),
          arr.elemSize(),
          from,
          1,
          to.withByteOffset(arr.offset() * arr.elemSize()),
          1);
  }

  public static void free(Pointer...pointers) {
    JCublas.cublasInit();
    for(Pointer arr : pointers) {
      int err = JCublas.cublasFree(arr);
//      assert (err == cublasStatus.CUBLAS_STATUS_SUCCESS):"Not successfully freed device memory";
      cudaCount--;
    }
  }

  public static void axpy(double alpha, DMatrix x, DMatrix y) {
//    DataTypeValidation.assertDouble(x,y);
    JCublas.cublasInit();
    CUDAMatrix xC = (CUDAMatrix) x;
    CUDAMatrix yC = (CUDAMatrix) y;
    Pointer xCPointer = (xC.persist())?xC.pointer():alloc(xC);
    Pointer yCPointer = (yC.persist())?yC.pointer():alloc(yC);
    JCublas.cublasDaxpy(xC.length(), alpha, xCPointer, 1, yCPointer, 1);
    // TODO: do you need it?
    if(!y.persist())
      getData(yC,yCPointer,Pointer.to(yC.data()));

    if(!x.persist())
      free(xCPointer);
    if(!y.persist())
      free(yCPointer);
  }

  /** Fils B with A
   */
  public static DMatrix fillWithArray(DMatrix A, DMatrix B) {
    JCublas.cublasInit();
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "kFillArray");

    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;

    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);

    Pointer kernelParameters = Pointer.to(Pointer.to(cAPointer),
        Pointer.to(new int[]{cA.length()}),
        Pointer.to(cBPointer),
        Pointer.to(new int[]{cB.length()})
        );

    cuLaunchKernel(function, getGridDim(cB.length()), 1, 1, getBlockDim(cB.length()), 1, 1, 0, null, kernelParameters, null);
    
    if(!cB.persist()) {
      getData(cB,cBPointer,Pointer.to(cB.data()));
      free(cBPointer);
    }

    if(!cA.persist())
      free(cAPointer);
    
    return B;
  }

  /** Mul rows of B by elemets of column vector A
   */
  public static DMatrix mulRows(DMatrix A, DMatrix B) {
    JCublas.cublasInit();
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "kMulByColumnVector");

    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;

    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);

    Pointer kernelParameters = Pointer.to(Pointer.to(cAPointer),
        Pointer.to(new int[]{cB.columns()}),
        Pointer.to(cBPointer),
        Pointer.to(new int[]{cB.length()})
        );

    cuLaunchKernel(function, getGridDim(cB.length()), 1, 1, getBlockDim(cB.length()), 1, 1, 0, null, kernelParameters, null);
    
    if(!cB.persist()) {
      getData(cB,cBPointer,Pointer.to(cB.data()));
      free(cBPointer);
    }

    if(!cA.persist())
      free(cAPointer);
    
    return B;
  }
  
  /** Div rows of B by elemets of column vector A
   */
  public static DMatrix divRows(DMatrix A, DMatrix B) {
    JCublas.cublasInit();
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "kDivByColumnVector");

    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;

    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);

    Pointer kernelParameters = Pointer.to(Pointer.to(cAPointer),
        Pointer.to(new int[]{cB.columns()}),
        Pointer.to(cBPointer),
        Pointer.to(new int[]{cB.length()})
        );

    cuLaunchKernel(function, getGridDim(cB.length()), 1, 1, getBlockDim(cB.length()), 1, 1, 0, null, kernelParameters, null);
    
    if(!cB.persist()) {
      getData(cB,cBPointer,Pointer.to(cB.data()));
      free(cBPointer);
    }

    if(!cA.persist())
      free(cAPointer);
    
    return B;
  }
  
  public static DMatrix mul(DMatrix A, DMatrix B, DMatrix C) {

    JCublas.cublasInit();
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "kMul");


    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;
    CUDAMatrix cC = (CUDAMatrix) C;
   
    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);
    Pointer cCPointer = (cC.persist())?cC.pointer():alloc(cC);

    Pointer kernelParameters = Pointer.to(Pointer.to(cAPointer),
        Pointer.to(cBPointer),
        Pointer.to(cCPointer),
        Pointer.to(new int[]{cA.length()})
        );
    // get dimensions right
    cuLaunchKernel(function, getGridDim(cA.length()), 1, 1, getBlockDim(A.length()), 1, 1, 0, null, kernelParameters, null);
    
    if(!cC.persist()) {
      getData(cC,cCPointer,Pointer.to(cC.data()));
      free(cCPointer);
    }

    if(!cA.persist())
      free(cAPointer);

    if(!cB.persist())
      free(cBPointer);


    return C;
  }

  // performs x^0.5 for each element x of the matrix
  public static DMatrix sqrt(DMatrix A) {
    JCublas.cublasInit();
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "kSqrt");

    CUDAMatrix cA = (CUDAMatrix) A;
    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    
    Pointer kernelParameters = Pointer.to(Pointer.to(cAPointer),
        Pointer.to(new int[]{cA.length()})
        );
  
    cuLaunchKernel(function, getGridDim(cA.length()), 1, 1, getBlockDim(cA.length()), 1, 1, 0, null, kernelParameters, null);
    
    if(!cA.persist()) {
      getData(cA,cAPointer,Pointer.to(cA.data()));
      free(cAPointer);
    }

    return A;
  }
  
  // performs 1/x for each element x of the matrix
  public static DMatrix inverseElements(DMatrix A) {
    JCublas.cublasInit();
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "kInverseElements");

    CUDAMatrix cA = (CUDAMatrix) A;
    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    
    Pointer kernelParameters = Pointer.to(Pointer.to(cAPointer),
        Pointer.to(new int[]{cA.length()})
        );
  
    cuLaunchKernel(function, getGridDim(cA.length()), 1, 1, getBlockDim(cA.length()), 1, 1, 0, null, kernelParameters, null);
    
    if(!cA.persist()) {
      getData(cA,cAPointer,Pointer.to(cA.data()));
      free(cAPointer);
    }

    return A;
  }

  public static DMatrix tanh(DMatrix A, DMatrix B) {
    JCublas.cublasInit();
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "kTanh");

    CUDAMatrix cA = (CUDAMatrix) A;
    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    
    CUDAMatrix cB = (CUDAMatrix) B;
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);

    Pointer cALengthPointer = alloc(cA.length());
    
    Pointer kernelParameters = Pointer.to(Pointer.to(cAPointer),
        Pointer.to(cBPointer),
        cALengthPointer
        );
  
    cuLaunchKernel(function, getGridDim(cA.length()), 1, 1, getBlockDim(cA.length()), 1, 1, 0, null, kernelParameters, null);
    
    if(!cB.persist()) {
      getData(cB,cBPointer,Pointer.to(cB.data()));
      free(cBPointer);
    }

    if(!cA.persist()) {
      free(cAPointer);
    }

    free(cALengthPointer);

    return B;
  }
  
  public static DMatrix pow(DMatrix A, double v) {
    JCublas.cublasInit();
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "kPow");

    CUDAMatrix cA = (CUDAMatrix) A;
    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    
    Pointer kernelParameters = Pointer.to(Pointer.to(cAPointer),
        Pointer.to(new double[]{v}),
        Pointer.to(new int[]{cA.length()})
        );
  
    cuLaunchKernel(function, getGridDim(cA.length()), 1, 1, getBlockDim(cA.length()), 1, 1, 0, null, kernelParameters, null);
    
    if(!cA.persist()) {
      getData(cA,cAPointer,Pointer.to(cA.data()));
      free(cAPointer);
    }

    return A;
  }

  public static DMatrix cust_gemv(boolean ta, DMatrix A, DMatrix B, DMatrix C, double alpha, double beta, int start, int howMany) {
  //     DataTypeValidation.assertDouble(A,B,C);
    JCublas.cublasInit();

    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;
    CUDAMatrix cC = (CUDAMatrix) C;
   
    Pointer cAPointer = alloc(cA, start*cA.columns(), howMany*cA.columns());
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);
    Pointer cCPointer = (cC.persist())?cC.pointer():alloc(cC);

    char opA = ta?'t':'n';
    int m = howMany;
    int n = cA.columns();
    int lda = ta?m:n;

   JCublas.cublasDgemv(
       opA,
       n,
       m,
       alpha,
       cAPointer,
       lda,
       cBPointer,
       1,
       beta,
       cCPointer,
       1);

   
    if(!cC.persist()) {
      getData(cC,cCPointer,Pointer.to(cC.data()));
      free(cCPointer);
    }

//    if(!cA.persist())
      free(cAPointer);

    if(!cB.persist())
      free(cBPointer);
    
    return C;
  }
  
  public static DMatrix gemv(boolean ta, DMatrix A, DMatrix B, DMatrix C, double alpha, double beta) {
    JCublas.cublasInit();

    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;
    CUDAMatrix cC = (CUDAMatrix) C;
   
    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);
    Pointer cCPointer = (cC.persist())?cC.pointer():alloc(cC);

    char opA = ta?'t':'n';
    int m = cA.rows();
    int n = cA.columns();
    int lda = ta?m:n;

   JCublas.cublasDgemv(
       opA,
       n,
       m,
       alpha,
       cAPointer,
       lda,
       cBPointer,
       1,
       beta,
       cCPointer,
       1);

   
    if(!cC.persist()) {
      getData(cC,cCPointer,Pointer.to(cC.data()));
      free(cCPointer);
    }

    if(!cA.persist())
      free(cAPointer);

    if(!cB.persist())
      free(cBPointer);
    
    return C;
  }

  public static DMatrix gemm(boolean ta, boolean tb, DMatrix A, DMatrix B, DMatrix C,
      double alpha, double beta) {
    JCublas.cublasInit();
    
    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;
    CUDAMatrix cC = (CUDAMatrix) C;
   
    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);
    Pointer cCPointer = (cC.persist())?cC.pointer():alloc(cC);

    char opA = tb?'t':'n';
    char opB = ta?'t':'n';

    int m = cC.rows();
    int n = cC.columns();
    int k = ta?cA.rows():cA.columns();
  
    int lda = ta?m:k; // (m = A.rows, 
    int ldb = tb?k:n;
   

    JCublas.cublasDgemm(
        opA, //trans
        opB,
        n, // m
        m, // n
        k, //k,
        alpha,
        cBPointer, // A
        ldb, // lda
        cAPointer, // x
        lda, // ldb
        beta, // beta
        cCPointer, // y
        n); // incy
    
    if(!cC.persist()) {
      getData(cC,cCPointer,Pointer.to(cC.data()));
      free(cCPointer);
    }

    if(!cA.persist())
      free(cAPointer);

    if(!cB.persist())
      free(cBPointer);
    
    return C;
  }

  public static DMatrix copy(DMatrix A, DMatrix B) {
    JCublas.cublasInit();

    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;

    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);

    JCublas.cublasDcopy(A.length(), 
        cAPointer, 
        1, 
        cBPointer, 
        1);
    if(!cB.persist()) {
      getData(cB,cBPointer,Pointer.to(cB.data()));
      free(cBPointer);
    }

    if(!cA.persist())
      free(cAPointer);
    return B;
  }

  public static DMatrix scal(DMatrix A, double alpha) {
    JCublas.cublasInit();

    CUDAMatrix cA = (CUDAMatrix) A;

    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    
    JCublas.cublasDscal(A.length(), alpha, cAPointer, 1);
    
    if(!cA.persist()) {
      getData(cA,cAPointer,Pointer.to(cA.data()));
      free(cAPointer);
    }
    return A;
  }
}
