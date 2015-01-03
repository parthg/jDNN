package nn;

import java.util.Arrays;
import org.jblas.DoubleMatrix;

public abstract class Layer {
  DoubleMatrix w, b;

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

  public double[] getParameters() {
//    System.out.printf("Num of Parametrs in Layer = %d\n", this.getThetaSize());
    double[] params = new double[this.thetaSize];
    System.arraycopy(this.w.toArray(), 0, params, 0, this.wSize);
    System.arraycopy(this.b.toArray(), 0, params, this.wSize, this.bSize);
    return params;
  }

  public double[] getParamGradients(DoubleMatrix myInData, DoubleMatrix mygrad) {
    double[] paramGrads = new double[this.thetaSize];
  
    double[] mydW = (myInData.transpose().mmul(mygrad)).toArray();
    double[] mydB = mygrad.toArray();

    System.arraycopy(mydW, 0, paramGrads, 0, this.wSize);
    System.arraycopy(mydB, 0, paramGrads, this.wSize, this.bSize);
    return paramGrads;
  }

  public void setParameters(double[] params) {
    this.w = new DoubleMatrix(this.inSize, this.size, Arrays.copyOfRange(params, 0, this.wSize));
    this.b = new DoubleMatrix(1, this.size, Arrays.copyOfRange(params, this.wSize, params.length));
  }

  public void init(boolean rand, int _inSize, int outSize) {
    this.inSize = _inSize;
    if(rand) {
      this.w = DoubleMatrix.randn(this.inSize, outSize).muli(0.01);
      this.b = DoubleMatrix.ones(1, outSize).muli(-2.0);
    } 
    //TODO: Think that do we need to init any params to zero ? or this can be a good way to get rid of cleargrads methods
    //else
    //  this.w = DoubleMatrix.zeros(this.inSize, outSize);
    this.wSize = this.inSize * outSize;
    this.bSize = outSize;
    this.thetaSize = this.wSize + this.bSize;
  }

  public DoubleMatrix getWeights() {
    return this.w;
  }

  public DoubleMatrix getBiases() {
    return this.b;
  }

  public abstract DoubleMatrix applyNonLinearity(DoubleMatrix input);
  
  public abstract DoubleMatrix fProp(DoubleMatrix input);

  // return the grads of this layers. Call subsequently getParaGradients() to get parameters gradients (flattened).
  public abstract DoubleMatrix bProp(DoubleMatrix mydata, DoubleMatrix error);

  public int getWSize() {return this.wSize;}
  public int getBSize() {return this.bSize;}
  public int getThetaSize() {return this.thetaSize;}
}
