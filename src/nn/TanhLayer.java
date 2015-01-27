package nn;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import math.DMatrix;
import math.DMatrixFunctions;
import math.DMath;

public class TanhLayer extends Layer {
  
  public TanhLayer(int _size) {
    super(_size);
  }

  public String name() {
    return "tanh";
  }
  
  /** tanh f(x) = (1-exp(-2x)) / (1+exp(-2x))
   */
  public DMatrix applyNonLinearity(DMatrix data) {
    return DMatrixFunctions.tanh(data);
  }

  /** fprop
   */
  public DMatrix fProp(DMatrix input) {
//    System.out.printf("in fprop\n");
    DMatrix data = DMath.createMatrix(input.rows(), this.getSize());
    data.fillWithArray(this.b);
    data.addi(input.mmul( this.w));
    
    //TODO: IMPORTANT - if you want to take advantage of device, have sigmoid on device
//    data.copyDtoH();
//    data.close();
    return this.applyNonLinearity(data);
  }

  /** calculate the gradient based on the error and representation provided to it
   * df/dx = (1-f(x))*(1+f(x)) = 1 - f(x)^2
   */
  public DMatrix bProp(DMatrix data, DMatrix error) {
    //TODO: Probably you can have data copy once on the device 
    //TODO: Check where you want to fill error in case of batch.
//    return error.mul(data.mul(DMath.createOnesMatrix(data.rows(), data.columns()).addi(-1.0, data)));
    return error.mul(DMath.createOnesMatrix(data.rows(), data.columns()).subi(data.mul(data)));
  }
  
}
