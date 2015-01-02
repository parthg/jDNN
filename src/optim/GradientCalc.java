package optim;

import java.util.List;
import org.jblas.DoubleMatrix;
import models.Model;
import common.Sentence;
import common.Datum;

public abstract class GradientCalc implements Optimizable.ByGradientValue {

  Model model;
  List<Datum> data;
  double[] params;

  double cost;
  double[] grads;

  public GradientCalc(List<Datum> _data) {
    this.data = _data;
  }

  public int dataSize() {
    return this.data.size();
  }
  public void setModel(Model _model) { this.model = _model; this.params = this.model.getParameters();}

//  public void setData(List<Sentence> _s) {this.s = _s;}
  /** TODO:parth include the code from mallet/optim and then see the structure of passing the parameters and then updateing it. - ConjugateGradient
   */

  public void getParameters(double[] doubleArray) {
    doubleArray = new double[this.model.getThetaSize()];
    System.arraycopy(this.params, 0, doubleArray, 0, this.getNumParameters());
  }

  public int getNumParameters() { return this.params.length; }

  public double getParameter(int n) { return params [n]; };

  public void setParameters(double[] doubleArray) {
    assert (doubleArray.length == this.params.length);
    System.arraycopy(doubleArray, 0, this.params, 0, this.getNumParameters());
    this.model.setParameters(this.params);
  }
  public void setParameter(int n, double d) { 
    params[n] = d;
    this.model.setParameters(this.params);
  }

}
