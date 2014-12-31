package optim;

import org.jblas.DoubleMatrix;
import models.Model;
import common.Sentence;

public class BasicGradientCalc extends GradientCalc {

  // f - error 
  public double getValue () {
    DoubleMatrix s1_root = this.model.fProp(this.s.get(0));
    DoubleMatrix s2_root = this.model.fProp(this.s.get(1));

    return 0.5*Math.pow(s1_root.distance2(s2_root),2);
  }

  // df - gradient for this error
  public void getValueGradient (double[] buffer) {
    assert (buffer.length == this.model.getThetaSize());
    DoubleMatrix grads = DoubleMatrix.zeros(1, buffer.length);
    grads.addi(this.model.bProp(this.s.get(0), this.s.get(1)));
    grads.addi(this.model.bProp(this.s.get(1), this.s.get(0)));

    System.arraycopy(grads.toArray(), 0, buffer, 0, this.model.getThetaSize());
  }
}
