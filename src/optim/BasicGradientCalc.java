package optim;

import org.jblas.DoubleMatrix;
import models.Model;
import common.Sentence;

public class BasicGradientCalc implements Optimizable.ByGradientValue {

  Model model;
  Sentence s1, s2;
  double[] params;

  public BasicGradientCalc(Model _model, Sentence _s1, Sentence _s2) {
    this.model = _model;
    this.s1 = _s1;
    this.s2 = _s2;

    this.params = new double[this.model.getThetaSize()];
  }
 

  /** TODO:parth include the code from mallet/optim and then see the structure of passing the parameters and then updateing it. - ConjugateGradient
   */

  public void getParameters(double[] doubleArray) {
    doubleArray = new double[this.model.getThetaSize()];
    System.arraycopy(this.params, 0, doubleArray, 0, this.getNumParameters());
  }

  public int getNumParameters() { return this.params.length; }

  public double getParameter(int n) { return params [n]; };

  public void setParameters(double[] doubleArray) {
    // TODO: assert that size is equal
    System.arraycopy(doubleArray, 0, this.params, 0, this.getNumParameters());
  }
  public void setParameter(int n, double d) { params[n] = d; }

  // f - error 
  public double getValue () {
    this.model.setParameters(this.params);
    DoubleMatrix s1_root = this.model.fProp(this.s1);
    DoubleMatrix s2_root = this.model.fProp(this.s2);

    return 2*s1_root.distance2(s2_root);
  }

  // df - gradient for this error
  public void getValueGradient (double[] buffer) {
    this.model.clearModelGrads();
    this.model.bProp(this.s1, this.s2);
    this.model.bProp(this.s2, this.s1);

    double[] dF = this.model.getGradients();

    System.arraycopy(buffer, 0, dF, 0, dF.length);
  }
}
