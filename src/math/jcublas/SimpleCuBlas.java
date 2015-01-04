package math.jcublas;

import jcuda.*;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;

import math.DMatrix;
import math.CUDAMatrix;

public class SimpleCuBlas {
  public static void close() {
    JCublas.cublasShutdown();
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
