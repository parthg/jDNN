package optim;

import org.jblas.DoubleMatrix;
import models.Model;
import common.Sentence;

public class BasicGradientCalc implements Optimizable.ByGradientValue {

  Model model;
  Sentence s1, s2;
  double[] params;

/*  public BasicGradientCalc(Model _model, Sentence _s1, Sentence _s2) {
    this.model = _model;
    this.s1 = _s1;
    this.s2 = _s2;

    this.params = new double[this.model.getThetaSize()];
  }*/
 
  public void setModel(Model _model) { this.model = _model; this.params = this.model.getParameters();}

  public void setData(Sentence _s1, Sentence _s2) {this.s1 = _s1; this.s2 = _s2;}
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
    this.model.setParameters(this.params);
  }
  public void setParameter(int n, double d) { 
    params[n] = d;
    this.model.setParameters(this.params);
  }

  // f - error 
  public double getValue () {
    DoubleMatrix s1_root = this.model.fProp(this.s1);
    DoubleMatrix s2_root = this.model.fProp(this.s2);

    return 0.5*Math.pow(s1_root.distance2(s2_root),2);
  }

  // df - gradient for this error
  public void getValueGradient (double[] buffer) {
//    buffer = new double[this.model.getThetaSize()];
    // TODO: assert(buffer.length == this.model.getThetaSize());
    this.model.clearModelGrads();
    this.model.bProp(this.s1, this.s2);
    this.model.bProp(this.s2, this.s1);

//    double[] dF = this.model.getGradients();

    System.arraycopy(this.model.getParamGradients(), 0, buffer, 0, this.model.getThetaSize());
  }
}
