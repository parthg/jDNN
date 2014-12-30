package optim;

import org.jblas.DoubleMatrix;
import models.Model;
import common.Sentence;

public class NoiseGradientCalc extends GradientCalc {

  // f - error 
  public double getValue () {
    DoubleMatrix s1_root = this.model.fProp(this.s.get(0));
    DoubleMatrix s2_root = this.model.fProp(this.s.get(1));
    DoubleMatrix s3_root = this.model.fProp(this.s.get(2));

    double err = 0.5*Math.pow(s1_root.distance2(s2_root),2)-0.5*Math.pow(s1_root.distance2(s3_root),2);
//    double err = 0.5*((s1_root.sub(s2_root)).mul(s1_root.sub(s2_root))).sum() - 0.5*((s1_root.sub(s3_root)).mul(s1_root.sub(s3_root))).sum();
    return err;
  }

  // df - gradient for this error
  public void getValueGradient (double[] buffer) {
    // TODO: assert(buffer.length == this.model.getThetaSize());
    this.model.clearModelGrads();
    // df/dA = (A-B) - (A-N)
    this.model.bProp(this.s.get(0), this.s.get(1), true);
    this.model.bProp(this.s.get(0), this.s.get(2), false);

    // df/dB = (B-A)
    this.model.bProp(this.s.get(1), this.s.get(0), true);

    // df/dN = - (N-A)
    this.model.bProp(this.s.get(2), this.s.get(0), false);

    System.arraycopy(this.model.getParamGradients(), 0, buffer, 0, this.model.getThetaSize());
  }
}
