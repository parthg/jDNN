package math;

import random.RandomUtils;
import math.jblas.SimpleBlas;

public class BLASMatrix extends DMatrix {
  
  public BLASMatrix(int r, int c) {
    super(r, c);
  }
  
  public BLASMatrix(int r, int c, double[] d) {
    super(r, c, d);
  }

  public static DMatrix zeros(int r, int c) {
    return new BLASMatrix(r, c);
  }

  public static DMatrix ones(int r, int c) {
    DMatrix m = new BLASMatrix(r, c);
    for(int i=0; i<r*c; i++)
      m.put(i, 1.0);
    return m;
  }

  public static DMatrix randn(int r, int c) {
    DMatrix m = new BLASMatrix(r, c);
    for (int i = 0; i < r * c; i++)
      m.put(i, RandomUtils.nextGaussian());
    return m;
  }

  public DMatrix transpose() {
    return new BLASMatrix(this.columns, this.rows, this.data);
  }
  
  public DMatrix add(DMatrix other) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  
  public DMatrix addi(DMatrix other) {
    System.out.printf("Using jblas\n");
    SimpleBlas.axpy(other, this);
    return this;
  }

  public DMatrix add(double v) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  public DMatrix addi(double v) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix addi(double a, DMatrix other) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

/*  public DMatrix addMuli(DMatrix A, DMatrix B) {
    return null;
  }*/

  public DMatrix inv() {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix invi() {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix sqrt() {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix sqrti() {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  
  public DMatrix subi(DMatrix other) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  public DMatrix sub(DMatrix other) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix sub(double v) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  public DMatrix subi(double v) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix mul(DMatrix other) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  public DMatrix muli(DMatrix other) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix mul(double v) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  public DMatrix muli(double v) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  
  public DMatrix pow(double v) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  public DMatrix powi(double v) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix mmul(DMatrix other, DMatrix result) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  public DMatrix mmul(boolean tA, boolean tB, DMatrix other, DMatrix result) {
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
      DMatrix temp = new BLASMatrix(result.rows(), result.columns());
  //    if (other.columns == 1) {
 //       SimpleBlas.gemv(this, other, temp, 1.0, 0.0);
 //     } else {
        SimpleBlas.gemm(this, other, temp, 1.0, 0.0);
//      }
      SimpleBlas.copy(temp, result);
    } 
    else {
//      if (other.columns == 1) {
//        SimpleBlas.gemv(this, other, result, 1.0, 0.0);
//      } else {
        SimpleBlas.gemm(this, other, result, 1.0, 0.0);
//      }
    }
    return result;
  }
  
  public DMatrix mmul(boolean tA, boolean tB, DMatrix B) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  
  public DMatrix mmul(DMatrix other) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  
  public DMatrix mmuli(boolean tA, boolean tB, DMatrix B) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix mmuli(DMatrix other) {
    return mmul(false, false, other, this);
  }

  public DMatrix fillWithArray(DMatrix other) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix divRows(DMatrix colVector) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  
  public DMatrix divRowsi(DMatrix colVector) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix mulRows(DMatrix colVector) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
  
  public DMatrix mulRowsi(DMatrix colVector) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix vectorNorm() {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix rowNorms() {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }

  public DMatrix dotRows(DMatrix B) {
    throw new UnsupportedOperationException("UNIMPLEMENTED");
  }
}
