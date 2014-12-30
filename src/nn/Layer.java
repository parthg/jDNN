package nn;

import java.util.Arrays;
import org.jblas.DoubleMatrix;

public abstract class Layer {
  DoubleMatrix w, b;
  DoubleMatrix inData, data, grad, allInData, allData;
  DoubleMatrix dW, dB;

  int wSize, bSize;
  int thetaSize; // basically wSize+bSize
  int inSize, size;

  public int getSize() {
    return this.size;
  }
  public Layer(int _size) {
    this.size = _size;
    this.allData = DoubleMatrix.zeros(1, this.size);
    this.data = DoubleMatrix.zeros(1,this.size);
    this.grad = DoubleMatrix.zeros(1,this.size);
  }

  public double[] getParameters() {
    System.out.printf("Num of Parametrs in Layer = %d\n", this.getThetaSize());
    double[] params = new double[this.thetaSize];
    System.arraycopy(this.w.toArray(), 0, params, 0, this.wSize);
    System.arraycopy(this.b.toArray(), 0, params, this.wSize, this.bSize);
    return params;
  }

  public double[] getParamGradients() {
    double[] paramGrads = new double[this.thetaSize];
    System.arraycopy(this.dW.toArray(), 0, paramGrads, 0, this.wSize);
    System.arraycopy(this.dB.toArray(), 0, paramGrads, this.wSize, this.bSize);
    return paramGrads;
  }

  public void setParameters(double[] params) {
    this.w = new DoubleMatrix(this.inSize, this.size, Arrays.copyOfRange(params, 0, this.wSize));
    this.b = new DoubleMatrix(1, this.size, Arrays.copyOfRange(params, this.wSize, params.length));
  }

  public void init(boolean rand, int _inSize, int outSize) {
    this.inSize = _inSize;
    this.allInData = DoubleMatrix.zeros(1, this.inSize);
    if(rand) {
      this.w = DoubleMatrix.randn(this.inSize, outSize);
      this.b = DoubleMatrix.zeros(1, outSize);
    } 
    //TODO: Think that do we need to init any params to zero ? or this can be a good way to get rid of cleargrads methods
    //else
    //  this.w = DoubleMatrix.zeros(this.inSize, outSize);
    this.wSize = this.inSize * outSize;
    this.bSize = outSize;
    this.thetaSize = this.wSize + this.bSize;

    this.dW = DoubleMatrix.zeros(this.inSize, outSize);
    this.dB = DoubleMatrix.zeros(1, outSize);
    this.grad = DoubleMatrix.zeros(1, outSize);
  }

  public void loadData(DoubleMatrix _data) {
    this.data = _data;
    this.grad = DoubleMatrix.zeros(this.data.rows,this.data.columns);
  }

  public void setData(DoubleMatrix _data) {
    this.data = _data;
  }

  public DoubleMatrix getActivities() {
    return this.data;
  }

  public DoubleMatrix getWeights() {
    return this.w;
  }

  public DoubleMatrix getBiases() {
    return this.b;
  }

  public DoubleMatrix getWeightGrads() {
    return this.dW;
  }

  public DoubleMatrix getBiasGrads() {
    return this.dB;
  }

  public DoubleMatrix getGradients() {
    return this.grad;
  }

  public void clearGrads() {
    this.grad = DoubleMatrix.zeros(1, this.size);
    this.dW = DoubleMatrix.zeros(this.inSize, this.size);
    this.dB = DoubleMatrix.zeros(1, this.size);
  }

/*  public void clearData() {
    this.data = DoubleMatrix.zeros(1, this.size);
    this.allData = DoubleMatrix.zeros(1, this.size);
    this.allInData = DoubleMatrix.zeros(1, this.inSize);
  }*

  /** accumulates the weight and bias gradient values based on current gradients.
   */
  public void accumulateGradients(boolean add) {
//    System.out.printf("All In data for this Sentence\n");
//    this.allInData.print();
    if(add) {
      dW.addi(this.inData.transpose().mmul(this.grad));
      dB.addi(this.grad);
    }
    else {
      dW.subi(this.inData.transpose().mmul(this.grad));
      dB.subi(this.grad);
    }
  }
  public abstract void applyNonLinearity();
  
  public abstract void fProp(DoubleMatrix input);

  public abstract void bProp(DoubleMatrix error);

  public int getWSize() {return this.wSize;}
  public int getBSize() {return this.bSize;}
  public int getThetaSize() {return this.thetaSize;}
}
