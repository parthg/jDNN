package optim;

import java.util.List;
import org.jblas.DoubleMatrix;
import models.Model;
import common.Sentence;

public abstract class GradientCalc implements Optimizable.ByGradientValue {

  Model model;
  List<Sentence> s;
  double[] params;

/*  public BasicGradientCalc(Model _model, Sentence _s1, Sentence _s2) {
    this.model = _model;
    this.s1 = _s1;
    this.s2 = _s2;

    this.params = new double[this.model.getThetaSize()];
  }*/
 
  public void setModel(Model _model) { this.model = _model; this.params = this.model.getParameters();}

  public void setData(List<Sentence> _s) {this.s = _s;}
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

/*  // f - error 
  public abstract double getValue ();

  // df - gradient for this error
  public abstract void getValueGradient (double[] buffer);*/
}
