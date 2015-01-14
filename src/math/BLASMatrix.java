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
    return null;
  }
  public DMatrix addi(DMatrix other) {
    System.out.printf("Using jblas\n");
    SimpleBlas.axpy(other, this);
    return this;
  }

  public DMatrix add(double v) {
    return null;
  }
  public DMatrix addi(double v) {
    return null;
  }

  public DMatrix addi(double a, DMatrix other) {
    return null;
  }

/*  public DMatrix addMuli(DMatrix A, DMatrix B) {
    return null;
  }*/

  public DMatrix subi(DMatrix other) {
    return null;
  }
  public DMatrix sub(DMatrix other) {
    return null;
  }

  public DMatrix mul(DMatrix other) {
    System.err.printf("TODO\n\n");
    return null;
  }
  public DMatrix muli(DMatrix other) {
    System.err.printf("TODO\n\n");
    return null;
  }

  public DMatrix mul(double v) {
    return null;
  }
  public DMatrix muli(double v) {
    return null;
  }
  
  public DMatrix pow(double v) {
    return null;
  }
  public DMatrix powi(double v) {
    return null;
  }

  public DMatrix mmul(DMatrix other, DMatrix result) {
    return null;
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
    System.err.printf("TODO\n\n");
    return null;
  }
  
  public DMatrix mmul(DMatrix other) {
    System.err.printf("TODO\n\n");
    return null;
  }
  
  public DMatrix mmuli(boolean tA, boolean tB, DMatrix B) {
    System.err.printf("TODO\n\n");
    return null;
  }

  public DMatrix mmuli(DMatrix other) {
    return mmul(false, false, other, this);
  }

  public DMatrix fillWithArray(DMatrix other) {
    return null;
  }
  public DMatrix divRows(DMatrix colVector) {
    return null;
  }
  public DMatrix divRowsi(DMatrix colVector) {
    return null;
  }
}
