package math.jcublas;

//import jcuda.*;
//import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;

import static jcuda.driver.JCudaDriver.*;
import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

import math.DMatrix;
import math.CUDAMatrix;

public class SimpleCuBlas {
  public static final int THREADS_PER_BLOCK = 1024;
  public static void close() {
    JCublas.cublasShutdown();
  }

  public static int getGridDim(int n) {
    return (n/THREADS_PER_BLOCK)+1;
  }
  public static int getBlockDim(int n) {
    return Math.min(n,THREADS_PER_BLOCK);
  }

  // cublasAlloc &  cublasSetVector H->D
  public static Pointer alloc(DMatrix m) {
    Pointer ret = new Pointer();
    Pointer toData = Pointer.to(m.data()).withByteOffset(m.offset() * m.elemSize());
    JCublas.cublasAlloc(
        m.length(),
        m.elemSize(),
        ret);
    JCublas.cublasSetVector(
        m.length(),
        m.elemSize(),
        toData,
        1,
        ret,
        1);
    return ret;
  }

  public static Pointer alloc(int m) {
    int[] arr = new int[1];
    arr[0] = m;
    Pointer ret = new Pointer();
    Pointer toData = Pointer.to(arr).withByteOffset(0 * 4);
    JCublas.cublasAlloc(
        1,
        4,
        ret);
    JCublas.cublasSetVector(
        1, // size of array
        4, // size of int
        toData,
        1,
        ret,
        1);
    return ret;
  }
  // cublasGetVector D->H
  public static void getData(DMatrix arr,Pointer from,Pointer to) {
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
    for(Pointer arr : pointers)
      JCublas.cublasFree(arr);
  }

  public static void axpy(double alpha, DMatrix x, DMatrix y) {
//    DataTypeValidation.assertDouble(x,y);
    JCublas.cublasInit();
    CUDAMatrix xC = (CUDAMatrix) x;
    CUDAMatrix yC = (CUDAMatrix) y;
    Pointer xCPointer = alloc(xC);
    Pointer yCPointer = alloc(yC);
    JCublas.cublasDaxpy(x.length(), alpha, xCPointer, 1, yCPointer, 1);
    getData(yC,yCPointer,Pointer.to(yC.data()));
    free(xCPointer,yCPointer);
  }

  public static DMatrix mul(DMatrix A, DMatrix B, DMatrix C) {

/*   CUDAMatrix cA = (CUDAMatrix) A;
   CUDAMatrix cB = (CUDAMatrix) B;
   CUDAMatrix cC = (CUDAMatrix) C;

   System.out.println(cA.data().length + " "+  cB.data().length + " " +cC.data().length);
//     int[] n = new int[1];
//     n[0] = A.length();
     cuInit(0);
     CUcontext pctx = new CUcontext();
     CUdevice dev = new CUdevice();
     cuDeviceGet(dev, 0);
     cuCtxCreate(pctx, 0, dev);
     
     CUmodule module = new CUmodule();
     cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
     CUfunction function = new CUfunction();
     cuModuleGetFunction(function, module, "kMul");

     CUdeviceptr a_dev = new CUdeviceptr();
     cuMemAlloc(a_dev, Sizeof.DOUBLE*A.length());
     cuMemcpyHtoD(a_dev, Pointer.to(cA.data()), Sizeof.DOUBLE*A.length());

     CUdeviceptr b_dev = new CUdeviceptr();
     cuMemAlloc(b_dev, Sizeof.DOUBLE*A.length());
     cuMemcpyHtoD(b_dev, Pointer.to(cB.data()), Sizeof.DOUBLE*A.length());

     CUdeviceptr c_dev = new CUdeviceptr();
     cuMemAlloc(c_dev, Sizeof.DOUBLE*A.length());

     Pointer kernelParameters = Pointer.to(
                                Pointer.to(a_dev),
                                Pointer.to(b_dev),
                                Pointer.to(c_dev));


     cuLaunchKernel(function, 1, 1, 1, A.length(), 1, 1, 0, null, kernelParameters, null);



     cuMemcpyDtoH(Pointer.to(cC.data()), c_dev, Sizeof.DOUBLE*A.length());
    
     JCuda.cudaFree(a_dev);
     JCuda.cudaFree(b_dev);
     JCuda.cudaFree(c_dev);*/


    JCublas.cublasInit();
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "kMul");

//    System.out.printf("Loaded\n");

    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;
    CUDAMatrix cC = (CUDAMatrix) C;
   
    Pointer cAPointer = alloc(cA);
    Pointer cBPointer = alloc(cB);
    Pointer cCPointer = alloc(cC);

    Pointer kernelParameters = Pointer.to(Pointer.to(cAPointer),
        Pointer.to(cBPointer),
        Pointer.to(cCPointer),
        Pointer.to(new int[]{cA.length()})
        );
    // get dimensions right
    cuLaunchKernel(function, getGridDim(cA.length()), 1, 1, getBlockDim(A.length()), 1, 1, 0, null, kernelParameters, null);
    getData(cC,cCPointer,Pointer.to(cC.data()));
    free(cAPointer,cBPointer,cCPointer);

    return C;
  }

  public static DMatrix gemv(DMatrix A, DMatrix B, DMatrix C, double alpha, double beta) {
  //     DataTypeValidation.assertDouble(A,B,C);
   JCublas.cublasInit();

   CUDAMatrix cA = (CUDAMatrix) A;
   CUDAMatrix cB = (CUDAMatrix) B;
   CUDAMatrix cC = (CUDAMatrix) C;
   
   Pointer cAPointer = alloc(cA);
   Pointer cBPointer = alloc(cB);
   Pointer cCPointer = alloc(cC);

   JCublas.cublasDgemv(
       'N',
       A.rows(),
       A.columns(),
       alpha,
       cAPointer,
       A.rows(),
       cBPointer,
       1,
       beta,
       cCPointer,
       1);

   getData(cC,cCPointer,Pointer.to(cC.data()));
   free(cAPointer,cBPointer,cCPointer);
   
   return C;
  }


  public static DMatrix gemm(DMatrix A, DMatrix B, DMatrix C,
      double alpha, double beta) {
//    DataTypeValidation.assertDouble(A,B,C);
    JCublas.cublasInit();
    
    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;
    CUDAMatrix cC = (CUDAMatrix) C;
    
    Pointer cAPointer = alloc(cA);
    Pointer cBPointer = alloc(cB);
    Pointer cCPointer = alloc(cC);

    JCublas.cublasDgemm(
        'n', //trans
        'n',
        C.rows(), // m
        C.columns(), // n
        A.columns(), //k,
        alpha,
        cAPointer, // A
        A.rows(), // lda
        cBPointer, // x
        B.rows(), // ldb
        beta, // beta
        cCPointer, // y
        C.rows()); // incy
    
    getData(cC,cCPointer,Pointer.to(cC.data()));
    free(cAPointer,cBPointer,cCPointer);
    return C;
  }

  public static DMatrix copy(DMatrix A, DMatrix B) {
    JCublas.cublasInit();

    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;

    Pointer cAPointer = alloc(cA);
    Pointer cBPointer = alloc(cB);

    JCublas.cublasDcopy(A.length(), 
        cAPointer, 
        1, 
        cBPointer, 
        1);
    getData(cB,cBPointer,Pointer.to(cB.data()));
    free(cAPointer, cBPointer);
    return B;
  }

  public static DMatrix scal(DMatrix A, double alpha) {
    JCublas.cublasInit();

    CUDAMatrix cA = (CUDAMatrix) A;

    Pointer cAPointer = alloc(cA);

    JCublas.cublasDscal(A.length(), alpha, cAPointer, 1);
    getData(cA,cAPointer,Pointer.to(cA.data()));
    free(cAPointer);
    return A;
  }

}
