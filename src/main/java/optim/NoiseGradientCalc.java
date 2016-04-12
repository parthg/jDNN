package optim;

import math.DMath;
import math.DMatrix;
import models.Model;
import common.Sentence;
import common.Datum;
import common.Batch;
import parallel.Parallel;
import common.Metric;

import java.util.List;

public class NoiseGradientCalc extends GradientCalc {

  public NoiseGradientCalc(Batch _batch) {
    super(_batch);
  }
  // f - error 
  public void testStats (Batch _batch) {
//    System.out.printf("calling getValue()\n");
    double err = 0.0;
    double batchError = 0.0;
    int nDP = 1;
    if(System.getProperty("use_cuda").equals("true")) {
        DMatrix A = this.model.fProp(_batch.data());
        DMatrix B = this.model.fProp(_batch.pos());
        DMatrix N = this.model.fProp(_batch.neg());

        DMatrix ARep = this.prepareSumMatrix(A, _batch.dataWordsArray());
        DMatrix BRep = this.prepareSumMatrix(B, _batch.posWordsArray());
        DMatrix NRep = this.prepareSumMatrix(N, _batch.negWordsArray());

        ARep.copyHtoD();
        BRep.copyHtoD();
        NRep.copyHtoD();


        double negPart = 0.5*(double)(((ARep.sub(BRep)).mul(ARep.sub(BRep))).sumColumns().sumRows().get(0));
        double posPart = 0.5*(double)(((ARep.sub(NRep)).mul(ARep.sub(NRep))).sumColumns().sumRows().get(0));
        
        batchError = (posPart - negPart)/(_batch.nSamples()*_batch.size());


      this.testLoss = batchError;
      
      DMatrix ARepNorm = ARep.vectorNorm();
      DMatrix BRepNorm = BRep.vectorNorm();
      this.testMRR = Metric.mrr(ARepNorm.mmul(false, true, BRepNorm));
        ARep.close();
        BRep.close();
        NRep.close();
    }
    // make it parallel
    else {
      final MonoNoiseCost costComputer = new MonoNoiseCost(this.model);
      Parallel.For(_batch.listData(), new Parallel.Operation<Datum>() {
        public void perform(int index, Datum datum) {
          try {
            costComputer.computeCost(datum);
          } catch (Exception e) {
            System.err.println(e.getMessage());
          }
        }
      });
      err = costComputer.getCost();
      this.testLoss = err/_batch.size();
    }
  }
  
  
  // f - error 
  public double getValue () {
//    System.out.printf("calling getValue()\n");
    double err = 0.0;
    double batchError = 0.0;
    int nDP = 1;
    if(System.getProperty("use_cuda").equals("true")) {
        DMatrix A = this.model.fProp(this.batch.data());
        DMatrix B = this.model.fProp(this.batch.pos());
        DMatrix N = this.model.fProp(this.batch.neg());

        DMatrix ARep = this.prepareSumMatrix(A, this.batch.dataWordsArray());
        DMatrix BRep = this.prepareSumMatrix(B, this.batch.posWordsArray());
        DMatrix NRep = this.prepareSumMatrix(N, this.batch.negWordsArray());

        ARep.copyHtoD();
        BRep.copyHtoD();
        NRep.copyHtoD();


        double negPart = 0.5*(double)(((ARep.sub(BRep)).mul(ARep.sub(BRep))).sumColumns().sumRows().get(0));
        double posPart = 0.5*(double)(((ARep.sub(NRep)).mul(ARep.sub(NRep))).sumColumns().sumRows().get(0));
        
        batchError = (posPart - negPart)/(this.batch.nSamples()*this.batch.size());

        ARep.close();
        BRep.close();
        NRep.close();

      return batchError;
    }
    // make it parallel
    else {
      final MonoNoiseCost costComputer = new MonoNoiseCost(this.model);
      Parallel.For(this.batch.listData(), new Parallel.Operation<Datum>() {
        public void perform(int index, Datum datum) {
          try {
            costComputer.computeCost(datum);
          } catch (Exception e) {
            System.err.println(e.getMessage());
          }
        }
      });
      err = costComputer.getCost();
      return err/this.batch.size();
    }
  }

  // df - gradient for this error
  // TODO: inefficient as it fProps several times
  public void getValueGradient (double[] buffer) {
//    System.out.printf("Calling getGradient()\n");
    assert (buffer.length == this.model.getThetaSize());
    // initialise grads with persist = true
    DMatrix grads = DMath.createZerosMatrix(1, buffer.length);
    grads.copyHtoD();
    try {
      int nDP = 1;
      int nSamples = this.batch.nSamples();
      if(System.getProperty("use_cuda").equals("true")) {
        
        DMatrix A = this.model.fProp(this.batch.data());
        DMatrix B = this.model.fProp(this.batch.pos());
        DMatrix N = this.model.fProp(this.batch.neg());

        A.copyHtoD();
        B.copyHtoD();
        N.copyHtoD();


        DMatrix ARep = this.prepareSumMatrix(A, this.batch.dataWordsArray());
        DMatrix BRep = this.prepareSumMatrix(B, this.batch.posWordsArray());
        DMatrix NRep = this.prepareSumMatrix(N, this.batch.negWordsArray());

        DMatrix AB = this.prepareErrorMatrix(ARep.sub(BRep), this.batch.dataWordsArray(), A.rows());
        DMatrix BA = this.prepareErrorMatrix(BRep.sub(ARep), this.batch.posWordsArray(), B.rows());
        DMatrix AN = this.prepareErrorMatrix(ARep.sub(NRep), this.batch.dataWordsArray(), A.rows());
        DMatrix NA = this.prepareErrorMatrix(NRep.sub(ARep), this.batch.negWordsArray(), N.rows());

        AB.copyHtoD();
        BA.copyHtoD();
        AN.copyHtoD();
        NA.copyHtoD();

        // df/dA = (A-N) - (A-B)
        DMatrix tempGrad = this.model.bProp(this.batch.data(), A, AN);
        grads.addi(tempGrad);
        tempGrad.close();

        tempGrad = this.model.bProp(this.batch.data(), A, AB);
        grads.subi(tempGrad);
        tempGrad.close();

        // df/dB = -(B-A)
        tempGrad = this.model.bProp(this.batch.pos(), B, BA);
        grads.subi(tempGrad);
        tempGrad.close();

        // df/dN = (N-A)
        tempGrad = this.model.bProp(this.batch.neg(), N, NA);
        grads.addi(tempGrad);
        tempGrad.close();

        grads.muli(1.0/(nSamples*this.batch.size()));

        A.close();
        B.close();
        N.close();

        AB.close();
        BA.close();
        AN.close();
        NA.close();
      
        grads.copyDtoH();
      grads.close();
      } 
      else {
        // parallelise this
        final MonoNoiseCost costComputer = new MonoNoiseCost(this.model);
        Parallel.For(this.batch.listData(), new Parallel.Operation<Datum>() {
          public void perform(int index, Datum datum) {
            try {
              costComputer.computeGrad(datum);
            } catch (Exception e) {
              System.err.println(e.getMessage());
            }
          }
        });
        grads.addi(costComputer.getGrads());
        grads.muli(1.0/this.batch.size());
      }
    } finally {
      grads.close();
    }
    System.arraycopy(grads.toArray(), 0, buffer, 0, this.model.getThetaSize());
    // now close the grads because it was initialised with persist = true --> grads.close();
  }

  public DMatrix prepareSumMatrix(DMatrix mat, int[] words) {
    DMatrix newMat = DMath.createMatrix(words.length, mat.columns());
    int start = 0;
    for(int i=0; i<words.length; i++) {
      DMatrix sum = mat.sumRows(start, words[i]);
      newMat.fillRow(i, sum);
      start+=words[i];
    }
    return newMat;
  }

  public DMatrix prepareErrorMatrix(DMatrix mat, int[] words, int totRows) {
    DMatrix newMat = DMath.createMatrix(totRows, mat.columns());
    int start = 0;
    for(int i=0; i<words.length; i++) {
      newMat.fillRow(start, words[i], mat.getRow(i));
      start+=words[i];
    }
    return newMat;
  }
}

