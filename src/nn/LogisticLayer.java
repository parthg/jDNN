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
    DMatrix data = DMath.createMatrix(input.rows(), this.getSize());
    data.fillWithArray(this.b);
    data.addi(input.mmul( this.w));
//    data.print();
    
    //TODO: IMPORTANT - if you want to take advantage of device, have sigmoid on device
//    data.copyDtoH();
//    data.close();
    return this.applyNonLinearity(data);
  }

  /** calculate the gradient based on the error and representation provided to it
   */
  public DMatrix bProp(DMatrix data, DMatrix error) {
    //TODO: Probably you can have data copy once on the device 
    //TODO: Check where you want to fill error in case of batch.
//    return error.mul(data.mul(DMath.createOnesMatrix(data.rows(), data.columns()).addi(-1.0, data)));
    return error.mul(data.mul(DMath.createOnesMatrix(data.rows(), data.columns()).subi(data)));
//    return error.mul(data.mul((data.mul(-1)).add(1)));
  }
}
