package math.jblas;

import org.jblas.JavaBlas;

import math.DMatrix;

public class SimpleBlas {
  public static DMatrix axpy(DMatrix other, DMatrix result) {
//    SimpleBlas.axpy(1.0, other, result);
    JavaBlas.raxpy(other.length(), 1.0, other.data(), 0, 1, result.data(), 0, 1);
    return result;
  }
}
