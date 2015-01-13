package optim;

import java.util.List;
import models.Model;
import common.Sentence;
import common.Datum;
import common.Batch;
import cc.mallet.optimize.Optimizable;

public abstract class GradientCalc implements Optimizable.ByGradientValue {

  Model model;
  Batch batch;
  double[] params;

  double cost;
  double[] grads;

  public GradientCalc(Batch _batch) {
    this.batch = _batch;
  }

  public int dataSize() {
    return this.batch.size();
  }
  public void setModel(Model _model) { this.model = _model; this.params = this.model.getParameters();}

  public void getParameters(double[] doubleArray) {
    assert doubleArray.length==this.getNumParameters();
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
