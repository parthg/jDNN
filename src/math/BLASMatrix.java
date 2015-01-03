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
