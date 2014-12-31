package nn;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class LogisticLayer extends Layer {
  
  public LogisticLayer(int _size) {
    super(_size);
  }
  
  /** sigmoid f(x) = 1/ (1+ exp(-x))
   */
  public DoubleMatrix applyNonLinearity(DoubleMatrix data) {
    return ((MatrixFunctions.exp(data.mul(-1))).addi(1)).rdivi(1);
  }

  /** fprop
   */
  public DoubleMatrix fProp(DoubleMatrix input) {
    DoubleMatrix data = input.mmul(this.w).addRowVector(this.b);
    return this.applyNonLinearity(data);
  }

  /** calculate the gradient based on the error and representation provided to it
   */
  public DoubleMatrix bProp(DoubleMatrix data, DoubleMatrix error) {
    return error.mul(data.mul((data.mul(-1)).add(1)));
  }
}
