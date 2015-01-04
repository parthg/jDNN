package nn;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import math.DMatrix;
import math.DMatrixFunctions;
import math.DMath;

public class LogisticLayer extends Layer {
  
  public LogisticLayer(int _size) {
    super(_size);
  }
  
  /** sigmoid f(x) = 1/ (1+ exp(-x))
   */
  public DMatrix applyNonLinearity(DMatrix data) {
    return DMatrixFunctions.sigmoid(data);
//    return ((MatrixFunctions.exp(data.mul(-1))).addi(1)).rdivi(1);
  }

  /** fprop
   */
  public DMatrix fProp(DMatrix input) {
    DMatrix data = DMath.createMatrix(1, this.getSize());
    data.addi(this.b);
    data.addMuli(input, this.w);
//    DMatrix data = input.mmul(this.w).addRowVector(this.b);
    return this.applyNonLinearity(data);
  }

  /** calculate the gradient based on the error and representation provided to it
   */
  public DMatrix bProp(DMatrix data, DMatrix error) {
    return error.mul(data.mul(DMath.createOnesMatrix(data.rows(), data.columns()).addi(-1.0, data)));
//    return error.mul(data.mul((data.mul(-1)).add(1)));
  }
}
