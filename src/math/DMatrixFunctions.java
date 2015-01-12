package math;

public class DMatrixFunctions {
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
}
