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

}
