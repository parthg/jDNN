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
  // TODO: inefficient as it fProps several times
  public void getValueGradient (double[] buffer) {
    assert (buffer.length == this.model.getThetaSize());
    DoubleMatrix grads = DoubleMatrix.zeros(1, buffer.length);
    // df/dA = (A-B) - (A-N)
    grads.addi(this.model.bProp(this.s.get(0), this.s.get(1)));
    grads.subi(this.model.bProp(this.s.get(0), this.s.get(2)));

    // df/dB = (B-A)
    grads.addi(this.model.bProp(this.s.get(1), this.s.get(0)));

    // df/dN = - (N-A)
    grads.subi(this.model.bProp(this.s.get(2), this.s.get(0)));

    System.arraycopy(grads.toArray(), 0, buffer, 0, this.model.getThetaSize());
  }
}
