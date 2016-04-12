package math.jblas;

import org.jblas.JavaBlas;
import org.jblas.NativeBlas;

import math.DMatrix;

public class SimpleBlas {
  public static DMatrix axpy(DMatrix other, DMatrix result) {
//    SimpleBlas.axpy(1.0, other, result);
    JavaBlas.raxpy(other.length(), 1.0, other.data(), 0, 1, result.data(), 0, 1);
    return result;
  }

  public static DMatrix gemm(DMatrix a, DMatrix b, DMatrix c, double alpha, double beta) {
    NativeBlas.dgemm('N', 'N', c.rows(), c.columns(), a.columns(), alpha, a.data(), 0,
        a.rows(), b.data(), 0, b.rows(), beta, c.data(), 0, c.rows());
    return c;
  }

  public static DMatrix copy(DMatrix x, DMatrix y) {
    //NativeBlas.dcopy(x.length, x.data, 0, 1, y.data, 0, 1);
    JavaBlas.rcopy(x.length(), x.data(), 0, 1, y.data(), 0, 1);
    return y;
  }

  public static DMatrix fillWithArray(DMatrix a, DMatrix dest) {
    int m = a.length();
    for(int i=0; i<dest.length(); i++) {
      dest.put(i, a.get(i%m));
    }
    return dest;
  }
}
