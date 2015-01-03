package math;

import math.jcublas.SimpleCuBlas;
import random.RandomUtils;

public class CUDAMatrix extends DMatrix {
  public CUDAMatrix(int r, int c) {
    super(r, c);
  }

  public CUDAMatrix(int r, int c, double[] d) {
    super(r, c, d);
  }

  public static DMatrix zeros(int r, int c) {
    return new CUDAMatrix(r, c);
  }

  public static DMatrix ones(int r, int c) {
    DMatrix m = new CUDAMatrix(r, c);
    for(int i=0; i<r*c; i++)
      m.put(i, 1.0);
    return m;
  }

  public static DMatrix randn(int r, int c) {
    DMatrix m = new CUDAMatrix(r, c);
    for (int i = 0; i < r * c; i++)
      m.put(i, RandomUtils.nextGaussian());
    return m;
  }
  
  public DMatrix transpose() {
    return new CUDAMatrix(this.columns, this.rows, this.data);
  }
  
  // y = alpha*x+y
  public DMatrix add(DMatrix other) {
    DMatrix m = new CUDAMatrix(this.rows, this.columns, this.data());
    SimpleCuBlas.axpy(1.0, other, m);
    return m;
  }
  public DMatrix addi(DMatrix other) {
    System.out.printf("Using cuda blas\n");
    SimpleCuBlas.axpy(1.0, other, this);
    return this;
  }

  public DMatrix mul(DMatrix other) {
    return null;
  }

  public DMatrix muli(DMatrix other) {
    return null;
  }

  public DMatrix mul(double v) {
    return null;
  }
  public DMatrix muli(double v) {
    return null;
  }

  public DMatrix mmul(DMatrix other) {
    DMatrix m = new CUDAMatrix(this.rows(), this.columns());
    SimpleCuBlas.gemm(other, this, m, 1.0, 0.0);
    return m;
  }

  public DMatrix mmuli(DMatrix other, DMatrix result) {
    assert (this.columns()==other.rows());
    if (result.rows != rows || result.columns != other.columns) {
      if (result != this && result != other) {
        result.resize(this.rows, other.columns);
      } else {
        System.err.printf("Cannot resize result matrix because it is used in-place.\n\n");
      }
    }

    if (result == this || result == other) {
      /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
       * * allocating a temporary object on the side and copy the result later.
       * */
      DMatrix temp = new CUDAMatrix(result.rows(), result.columns());
      if (other.columns == 1) {
        SimpleCuBlas.gemv(this, other, temp, 1.0, 0.0);
      } else {
        SimpleCuBlas.gemm(this, other, temp, 1.0, 0.0);
      }
      SimpleCuBlas.copy(temp, result);
    } 
    else {
      if (other.columns == 1) {
        SimpleCuBlas.gemv(this, other, result, 1.0, 0.0);
      } else {
        SimpleCuBlas.gemm(this, other, result, 1.0, 0.0);
      }
    }

    return result;
  }
  public DMatrix mmuli(DMatrix other) {
    return mmuli(other, this);
  }

}
