package optim;

public static class BasicGradientCalc implements Optimizable.ByGradientValue {

  Model model;
  Sentence s1, s2;

  public BasicGradientCalc(Model _model, Sentence _s1, Sentence _s2) {
    this.model = _model;
    this.s1 = _s1;
    this.s2 = _s2;
  }
 

  /** TODO:parth include the code from mallet/optim and then see the structure of passing the parameters and then updateing it. - ConjugateGradient
   */
  double[] params = new double [1];

  public void getParameters(double[] doubleArray) {
    doubleArray [0] = params [0];
  }

  public int getNumParameters() { return 1; }

  public double getParameter(int n) { return params [0]; };

  public void setParameters(double[] doubleArray) {
    params [0] = doubleArray [0];
  }
  public void setParameter(int n, double d) { params[n] = d; }

  // f - error 
  public double getValue () {
    DoubleMatrix s1_root = this.model.output(this.s1);
    DoubleMatrix s2_root = this.model.output(this.s2);

    return 2*s1_root.distance2(s2_root);
  }

  // df - gradient for this error
  public void getValueGradient (double[] buffer) {
    this.model.clearModelGrads();
    this.model.backPropagate(this.s1, this.s2);
    this.model.backPropagate(this.s2, this.s1);

    double[] dF = this.model.getGradients();

    System.arraycopy(buffer, 0, dF, 0, dF.length);
  }
}
