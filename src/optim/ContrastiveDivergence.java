package optim;

import cc.mallet.optimize.Optimizer;
import cc.mallet.optimize.Optimizable;

public class ContrastiveDivergence implements Optimizer{
  public boolean optimize () {
    return false;
  }
  public boolean optimize (int numIterations) {
    return false;
  }
  public boolean isConverged() {
    return false;
  }
  public Optimizable getOptimizable() {
    return null;
  }
}
