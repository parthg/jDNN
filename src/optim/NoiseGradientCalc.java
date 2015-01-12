package optim;

//import org.jblas.DoubleMatrix;
import math.DMath;
import math.DMatrix;
import models.Model;
import common.Sentence;
import common.Datum;
import parallel.Parallel;

import java.util.List;

public class NoiseGradientCalc extends GradientCalc {

  public NoiseGradientCalc(List<Datum> _data) {
    super(_data);
  }
  // f - error 
  public double getValue () {
//    System.out.printf("calling getValue()\n");
    double err = 0.0;
    int nDP = 1;
    if(System.getProperty("use_cuda").equals("true")) {
      for(Datum d: this.data) {
//        DMatrix dataIn = this.model.dict().getRepresentation(d.getData());
//        DMatrix dataPos = this.model.dict().getRepresentation(d.getPos());

/*        DMatrix s1_root = this.model.fProp(d.getData());
        //s1_root.print();
        DMatrix s2_root = this.model.fProp(d.getPos());
//        s2_root.print();*/

        int nSamples = d.getNegSampleSize();
        nDP = nSamples;
        List<Sentence> neg = d.getNeg();
/*        double unitError = 0.0;
        for(int i=0; i<nSamples; i++) {

//          DMatrix dataNeg = this.model.dict().getRepresentation(neg.get(i));
          DMatrix s3_root = this.model.fProp(neg.get(i));

          unitError += -(0.5*s1_root.squaredDistance(s2_root))+(0.5*s1_root.squaredDistance(s3_root));
        }
//        System.out.printf("Error original = %.8f\t", unitError);*/
  
        DMatrix dataIn = this.model.dict().getRepresentation(d.getData());
        DMatrix dataPos = this.model.dict().getRepresentation(d.getPos());

        DMatrix s1_root = this.model.getRepresentation(dataIn);
//        s1_root.print();
        DMatrix s2_root = this.model.getRepresentation(dataPos);
//        s2_root.print();

        double unitError = 0.0;
        for(int i=0; i<nSamples; i++) {

          DMatrix dataNeg = this.model.dict().getRepresentation(neg.get(i));
          DMatrix s3_root = this.model.getRepresentation(dataNeg);

          unitError += -(0.5*s1_root.squaredDistance(s2_root))+(0.5*s1_root.squaredDistance(s3_root));
        }
//        System.out.printf("Batch Error = %.8f\n", unitError);
        err+=unitError;
      }
  //    double err = 0.5*((s1_root.sub(s2_root)).mul(s1_root.sub(s2_root))).sum() - 0.5*((s1_root.sub(s3_root)).mul(s1_root.sub(s3_root))).sum();
/*      System.gc(); System.gc();
      System.gc(); System.gc();
      System.gc(); System.gc();
      System.gc(); System.gc();*/
      return err/(nDP*this.dataSize());
    }
    // make it parallel
    else {
      final MonoNoiseCost costComputer = new MonoNoiseCost(this.model);
      Parallel.For(this.data, new Parallel.Operation<Datum>() {
        public void perform(int index, Datum datum) {
          try {
            costComputer.computeCost(datum);
          } catch (Exception e) {
            System.err.println(e.getMessage());
          }
        }
      });
      err = costComputer.getCost();
      return err/this.dataSize();
    }
  }

  // df - gradient for this error
  // TODO: inefficient as it fProps several times
  public void getValueGradient (double[] buffer) {
//    System.out.printf("Calling getGradient()\n");
    assert (buffer.length == this.model.getThetaSize());
    // initialise grads with persist = true
    DMatrix grads = DMath.createZerosMatrix(1, buffer.length, true);
    try {
      int nDP = 1;
      if(System.getProperty("use_cuda").equals("true")) {

        for(Datum d: this.data) {
          int nSamples = d.getNegSampleSize();
          nDP = nSamples;
          List<Sentence> neg = d.getNeg();
          
          try(DMatrix dataIn = this.model.dict().getRepresentation(d.getData());
            DMatrix dataPos = this.model.dict().getRepresentation(d.getPos());) {

            dataIn.copyHtoD();
            dataPos.copyHtoD();
          
            DMatrix A = this.model.fProp(dataIn);
            DMatrix B = this.model.fProp(dataPos);

            A.copyHtoD();
            B.copyHtoD();

            // TODO: This is the bottleneck for the whole batch together
            DMatrix AB = A.sumRows().sub(B.sumRows());
            DMatrix BA = B.sumRows().sub(A.sumRows());

            AB.copyHtoD();
            BA.copyHtoD();

            for(int i=0; i<nSamples; i++) {
              // df/dA = (A-N) - (A-B)
              
              try (DMatrix dataNeg = this.model.dict().getRepresentation(neg.get(i));) {
                
                dataNeg.copyHtoD();
                
                DMatrix N = this.model.fProp(dataNeg);
                N.copyHtoD();

                // TODO: This is the bottleneck for the whole batch together
                DMatrix AN = A.sumRows().sub(N.sumRows());
                DMatrix NA = N.sumRows().sub(A.sumRows());

                AN.copyHtoD();
                NA.copyHtoD();

                DMatrix tempGrad = this.model.bProp(dataIn, A, AN);
                grads.addi(tempGrad);
                tempGrad.close();

                tempGrad = this.model.bProp(dataIn, A, AB);
                grads.subi(tempGrad);
                tempGrad.close();

                // df/dB = -(B-A)
                tempGrad = this.model.bProp(dataPos, B, BA);
                grads.subi(tempGrad);
                tempGrad.close();

                // df/dN = (N-A)
                tempGrad = this.model.bProp(dataNeg, N, NA);
                grads.addi(tempGrad);
                tempGrad.close();

                N.close();

                AN.close();
                NA.close();
               
              }

            }

            A.close();
            B.close();

            AB.close();
            BA.close();
          }
        }

        grads.muli(1.0/(nDP*this.dataSize()));
      } 
      else {
        // parallelise this
        final MonoNoiseCost costComputer = new MonoNoiseCost(this.model);
        Parallel.For(this.data, new Parallel.Operation<Datum>() {
          public void perform(int index, Datum datum) {
            try {
              costComputer.computeGrad(datum);
            } catch (Exception e) {
              System.err.println(e.getMessage());
            }
          }
        });
        grads.addi(costComputer.getGrads());
        grads.muli(1.0/this.dataSize());
      }
    } finally {
      grads.copyDtoH();
      grads.close();
    }
    System.arraycopy(grads.toArray(), 0, buffer, 0, this.model.getThetaSize());
    // now close the grads because it was initialised with persist = true --> grads.close();
  }

}
