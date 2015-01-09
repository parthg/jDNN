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
//    System.out.printf("Allocationg memory\n");
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

  public static Pointer alloc(double[] arr) {
    Pointer ret = new Pointer();
    Pointer toData = Pointer.to(arr).withByteOffset(0 * Sizeof.DOUBLE);
    JCublas.cublasAlloc(
        arr.length,
        Sizeof.DOUBLE,
        ret);
    JCublas.cublasSetVector(
        arr.length, // size of array
        Sizeof.DOUBLE, // size of int
        toData,
        1,
        ret,
        1);
    return ret;
  }

  public static void updateData(Pointer ptr, double[] arr) {
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

  public static DMatrix fillWithArray(DMatrix A, DMatrix B) {
    JCublas.cublasInit();
    CUmodule module = new CUmodule();
    cuModuleLoad(module, "src/math/jcublas/cuda_kernels.ptx");
    CUfunction function = new CUfunction();
    cuModuleGetFunction(function, module, "kFillWithArray");

    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;

    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);

    Pointer kernelParameters = Pointer.to(Pointer.to(cAPointer),
        Pointer.to(new int[]{cA.length()}),
        Pointer.to(cBPointer),
        Pointer.to(new int[]{cB.length()})
        );

    cuLaunchKernel(function, getGridDim(B.length()), 1, 1, getBlockDim(B.length()), 1, 1, 0, null, kernelParameters, null);
    
    if(!cB.persist()) {
      System.out.println("Not persist so copying back");
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

//    System.out.printf("Loaded\n");

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

  public static DMatrix gemv(DMatrix A, DMatrix B, DMatrix C, double alpha, double beta) {
  //     DataTypeValidation.assertDouble(A,B,C);
    JCublas.cublasInit();

    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;
    CUDAMatrix cC = (CUDAMatrix) C;
   
    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);
    Pointer cCPointer = (cC.persist())?cC.pointer():alloc(cC);

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

//   getData(cC,cCPointer,Pointer.to(cC.data()));
//   free(cAPointer,cBPointer,cCPointer);
   
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

  public static DMatrix gemm(DMatrix A, DMatrix B, DMatrix C,
      double alpha, double beta) {
//    DataTypeValidation.assertDouble(A,B,C);
    JCublas.cublasInit();
    
    CUDAMatrix cA = (CUDAMatrix) A;
    CUDAMatrix cB = (CUDAMatrix) B;
    CUDAMatrix cC = (CUDAMatrix) C;
    
    Pointer cAPointer = (cA.persist())?cA.pointer():alloc(cA);
    Pointer cBPointer = (cB.persist())?cB.pointer():alloc(cB);
    Pointer cCPointer = (cC.persist())?cC.pointer():alloc(cC);

    System.out.printf("\n%d\n", cB.columns());
    
    JCublas.cublasDgemm(
        'n', //trans
        'n',
        cC.columns(), // m
        cC.rows(), // n
        cB.rows(), //k,
        alpha,
        cBPointer, // A
        cB.columns(), // lda
        cAPointer, // x
        cA.columns(), // ldb
        beta, // beta
        cCPointer, // y
        cC.columns()); // incy
    
//    getData(cC,cCPointer,Pointer.to(cC.data()));
//    free(cAPointer,cBPointer,cCPointer);
    if(!cC.persist()) {
      getData(cC,cCPointer,Pointer.to(cC.data()));
      free(cCPointer);
    }

//    System.out.printf("Aftre getting the data\n");
//    cC.print();

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
//    getData(cB,cBPointer,Pointer.to(cB.data()));
//    free(cAPointer, cBPointer);
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
//    getData(cA,cAPointer,Pointer.to(cA.data()));
//    free(cAPointer);
    return A;
  }

}
