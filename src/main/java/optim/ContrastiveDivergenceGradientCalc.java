package optim;

import common.Batch;

public class ContrastiveDivergenceGradientCalc extends GradientCalc {
  public ContrastiveDivergenceGradientCalc(Batch _batch) {
    super(_batch);
  }

  public void testStats (Batch _batch) {
    throw new UnsupportedOperationException("TODO");
  }
  
  public double getValue () {
    throw new UnsupportedOperationException("TODO");
  }

  public void getValueGradient (double[] buffer) {
    throw new UnsupportedOperationException("TODO");
  }
}
