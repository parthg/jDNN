package nn;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class LogisticLayer extends Layer {
  
  public LogisticLayer(int _size) {
    super(_size);
  }
  
  public void applyNonLinearity() {
    this.data = ((MatrixFunctions.exp(this.data.mul(-1))).addi(1)).rdivi(1);
  }

  /** this will change the current data of the network
   */
  public void fProp(DoubleMatrix input) {
    this.inData = input; 
//    this.allInData.addi(input);
    this.data = input.mmul(this.w).addRowVector(this.b);
    this.applyNonLinearity();
//    this.allData.addi(this.data);
  }

  /** calculate the gradient based on the error provided to it
   */
  public void bProp(DoubleMatrix error) {
    this.grad = error.mul(this.data.mul((this.data.mul(-1)).add(1)));
  }
}
