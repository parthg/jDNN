package optim;

import common.Batch;

public class ContrastiveDivergenceGradientCalc extends GradientCalc {
  public ContrastiveDivergenceGradientCalc(Batch _batch) {
    super(_batch);
  }

  public void testStats (Batch _batch) {
  }
  
  public double getValue () {
    return 0.0;
  }

  public void getValueGradient (double[] buffer) {
  }
}
