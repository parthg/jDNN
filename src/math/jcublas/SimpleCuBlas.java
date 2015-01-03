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
}
