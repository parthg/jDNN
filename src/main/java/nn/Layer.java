package nn;

import java.util.Arrays;
import math.DMath;
import math.DMatrix;
//import org.jblas.DoubleMatrix;

public abstract class Layer {
  DMatrix w, b;
  DMatrix data;

  double LAMBDA = 0.0; // default no regularization

  int wSize, bSize;
  int thetaSize; // basically wSize+bSize
  int inSize, size;


  public int getInSize() {
    return this.inSize;
  }

  public int getSize() {
    return this.size;
  }
  public Layer(int _size) {
    this.size = _size;
  }

  public void setRegularization(double _lambda) {
    this.LAMBDA = _lambda;
  }

  /** Returns Squared sum of weight parameters.
   * Mainly useful for refularization.
   */
  public double weightSquaredSum() {
    return this.w.mul(this.w).sumRows().sumColumns().get(0);
  }

  public double[] getParameters() {
    double[] params = new double[this.thetaSize];
    System.arraycopy(this.w.toArray(), 0, params, 0, this.wSize);
    System.arraycopy(this.b.toArray(), 0, params, this.wSize, this.bSize);
    return params;
  }
  
  /** Returns the weight parameters of the network with bias as zero.
   * Useful for regularization gradient.
   */
  public double[] getWeightOnlyParameters() {
    double[] params = new double[this.thetaSize];
    // copying only weights. Biases will be zero anyway.
    System.arraycopy(this.w.toArray(), 0, params, 0, this.wSize);
    return params;
  }

  public double[] getParamGradients(DMatrix myInData, DMatrix mygrad) {
    double[] paramGrads = new double[this.thetaSize];
    double[] mydW = (myInData.mmul(true, false, mygrad)).toArray();
    double[] mydB = mygrad.sumRows().toArray();

    System.arraycopy(mydW, 0, paramGrads, 0, this.wSize);
    System.arraycopy(mydB, 0, paramGrads, this.wSize, this.bSize);
    return paramGrads;
  }

  public void setParameters(double[] params) {
    this.w = DMath.createMatrix(this.inSize, this.size, Arrays.copyOfRange(params, 0, this.wSize));
    this.b = DMath.createMatrix(1, this.size, Arrays.copyOfRange(params, this.wSize, params.length));
    this.updateDeviceCopy();
  }

  public void init(boolean rand, int _inSize, int outSize) {
    this.init(rand, _inSize, outSize, 1.0, 1.0);
  }
  
  public void init(boolean rand, int _inSize, int outSize, double weightScale, double biasScale) {
    this.inSize = _inSize;
    if(rand) {
      this.w = DMath.createRandnMatrix(this.inSize, outSize).muli(weightScale);
      this.b = DMath.createOnesMatrix(1, outSize).muli(biasScale);
    }
    else {
      this.w = DMath.createMatrix(this.inSize, outSize);
      this.b = DMath.createZerosMatrix(1, outSize);
    } 
    //TODO: Think that do we need to init any params to zero ? or this can be a good way to get rid of cleargrads methods
    //else
    //  this.w = DoubleMatrix.zeros(this.inSize, outSize);
    this.wSize = this.inSize * outSize;
    this.bSize = outSize;
    this.thetaSize = this.wSize + this.bSize;
  }

  public DMatrix getWeights() {
    return this.w;
  }

  public DMatrix getBiases() {
    return this.b;
  }

  public void setWeights(DMatrix _w) {
    assert (_w.rows()==this.inSize && _w.columns()==this.size);
    this.w = _w;
  }

  public void setBiases(DMatrix _b) {
    assert(_b.rows() ==  1 && _b.columns() == this.size);
    this.b = _b;
  }

  public DMatrix getData() {
    return this.data;
  }

  public void setData(DMatrix _data) {
    this.data = _data;
  }

  public void copyHtoD() {
    this.w.copyHtoD();
    this.b.copyHtoD();
  }

  public void copyDtoH() {
    this.w.copyDtoH();
    this.b.copyDtoH();
  }

  public void clearDevice() {
    this.w.close();
    this.b.close();
  }

  public void updateDeviceCopy() {
    this.w.updateDeviceData();
    this.b.updateDeviceData();
  }

  public abstract String name();

  public abstract DMatrix applyNonLinearity(DMatrix input);
  
  public abstract DMatrix fProp(DMatrix input);

  // return the grads of this layers. Call subsequently getParaGradients() to get parameters gradients (flattened).
  public abstract DMatrix bProp(DMatrix mydata, DMatrix error);

  public int getWSize() {return this.wSize;}
  public int getBSize() {return this.bSize;}
  public int getThetaSize() {return this.thetaSize;}
}
