package math;

import math.jcublas.SimpleCuBlas;

public class DMatrixFunctions {
  public static boolean USE_CUDA = Boolean.parseBoolean(System.getProperty("use_cuda"));
  public DMatrixFunctions() {
    if(System.getProperty("use_cuda").equals("true"))
      USE_CUDA = true;
  }
  public static DMatrix expi(DMatrix x) {
    for (int i = 0; i < x.length(); i++)
      x.put(i, (double) Math.exp(x.get(i)));
    return x;
  }

  public static DMatrix sigmoid(DMatrix x) {
    for (int i = 0; i < x.length(); i++)
      x.put(i,(double)1.0/(1.0+Math.exp(-1.0*x.get(i))));
    return x;
  }
  
  public static DMatrix tanh(DMatrix x) {
    DMatrix m = DMath.createMatrix(x.rows(), x.columns(), x.toArray());
    if(USE_CUDA) {
      SimpleCuBlas.tanh(x, m);
    }
    else {
      for (int i = 0; i < x.length(); i++)
        m.put(i,(double)Math.tanh(x.get(i)));
    }
    return m;
  }
}
