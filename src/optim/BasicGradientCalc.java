package optim;

//import org.jblas.DoubleMatrix;
import math.DMath;
import math.DMatrix;
import models.Model;
import common.Sentence;
import common.Datum;
import common.Batch;

import java.util.List;

public class BasicGradientCalc extends GradientCalc {

  public BasicGradientCalc(Batch _batch) {
    super(_batch);
    System.err.printf("\n\nThis is still minimizer - FIX IT!\n\n");
//    System.exit(0);
  }

  public void testStats(Batch _testBatch) {
  }

  // f - error 
  public double getValue () {
    double err = 0.0;
    for(Datum d: this.batch.listData()) {
      DMatrix s1_root = this.model.fProp(d.getData());
      DMatrix s2_root = this.model.fProp(d.getPos());

      double unitError = 0.5*Math.pow(s1_root.distance2(s2_root),2);
      err+=unitError;
    }
    return err/this.dataSize();
  }

  // df - gradient for this error
  public void getValueGradient (double[] buffer) {
    assert (buffer.length == this.model.getThetaSize());
    DMatrix grads = DMath.createZerosMatrix(1, buffer.length);

    for(Datum d: this.batch.listData()) {
      grads.addi(this.model.bProp(d.getData(), d.getPos()));
      grads.addi(this.model.bProp(d.getPos(), d.getData()));
    }

    grads.muli(1.0/(this.dataSize()));
    System.arraycopy(grads.toArray(), 0, buffer, 0, this.model.getThetaSize());
  }
}
