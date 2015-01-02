package optim;

import org.jblas.DoubleMatrix;
import models.Model;
import common.Sentence;
import common.Datum;

import java.util.List;

public class NoiseGradientCalc extends GradientCalc {

  public NoiseGradientCalc(List<Datum> _data) {
    super(_data);
  }
  // f - error 
  public double getValue () {
    double err = 0.0;
    int nDP = 1;
    // make it parallel
    for(Datum d: this.data) {
      DoubleMatrix s1_root = this.model.fProp(d.getData());
      DoubleMatrix s2_root = this.model.fProp(d.getPos());
      int nSamples = d.getNegSampleSize();
      nDP = nSamples;
      List<Sentence> neg = d.getNeg();
      double unitError = 0.0;
      for(int i=0; i<nSamples; i++) {
        DoubleMatrix s3_root = this.model.fProp(neg.get(i));

        unitError += 0.5*Math.pow(s1_root.distance2(s2_root),2)-0.5*Math.pow(s1_root.distance2(s3_root),2);
      }
      err+=unitError;
    }
//    double err = 0.5*((s1_root.sub(s2_root)).mul(s1_root.sub(s2_root))).sum() - 0.5*((s1_root.sub(s3_root)).mul(s1_root.sub(s3_root))).sum();
    return err/(nDP*this.dataSize());
  }

  // df - gradient for this error
  // TODO: inefficient as it fProps several times
  public void getValueGradient (double[] buffer) {
    assert (buffer.length == this.model.getThetaSize());
    DoubleMatrix grads = DoubleMatrix.zeros(1, buffer.length);
    int nDP = 1;
    // parallelise this
    for(Datum d: this.data) {
      int nSamples = d.getNegSampleSize();
      nDP = nSamples;
      List<Sentence> neg = d.getNeg();

      for(int i=0; i<nSamples; i++) {
        // df/dA = (A-B) - (A-N)
        grads.addi(this.model.bProp(d.getData(), d.getPos()));
        grads.subi(this.model.bProp(d.getData(), neg.get(i)));

        // df/dB = (B-A)
        grads.addi(this.model.bProp(d.getPos(), d.getData()));

        // df/dN = - (N-A)
        grads.subi(this.model.bProp(neg.get(i), d.getData()));
      }
    }

    grads.muli(1.0/(nDP*this.dataSize()));

    System.arraycopy(grads.toArray(), 0, buffer, 0, this.model.getThetaSize());
  }

}
