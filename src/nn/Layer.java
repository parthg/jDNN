package nn;

import org.jblas.DoubleMatrix;

public abstract class Layer {
  DoubleMatrix w, b;
  DoubleMatrix inData, data, grad;
  DoubleMatrix dW, dB;

  int size;

  public int getSize() {
    return this.size;
  }
  public Layer(int _size) {
    this.size = _size;
    this.data = DoubleMatrix.zeros(1,this.size);
    this.grad = DoubleMatrix.zeros(1,this.size);
  }

  public void init(boolean rand, int inSize, int outSize) {
    if(rand) {
      this.w = DoubleMatrix.randn(inSize, outSize);
    } else
      this.w = DoubleMatrix.zeros(inSize, outSize);
    this.dW = DoubleMatrix.zeros(inSize, outSize);

    this.b = DoubleMatrix.zeros(1,outSize);
    this.dB = DoubleMatrix.zeros(1, outSize);

    this.grad = DoubleMatrix.zeros(1, outSize);
  }

  public void loadData(DoubleMatrix _data) {
    this.data = _data;
    this.grad = DoubleMatrix.zeros(this.data.rows,this.data.columns);
  }

  public DoubleMatrix getActivities() {
    return this.data;
  }

  public DoubleMatrix getWeights() {
    return this.w;
  }

  public DoubleMatrix getGradients() {
    return this.grad;
  }

  /** accumulates the weight and bias gradient values based on current gradients.
   */
  public void accumulateGradients() {
    dW.addi(this.inData.transpose().mmul(this.grad));
    dB.addi(this.grad);
  }
  public abstract void applyNonLinearity();
  
  public abstract void fProp(DoubleMatrix input);

  public abstract void bProp(DoubleMatrix error);

}
