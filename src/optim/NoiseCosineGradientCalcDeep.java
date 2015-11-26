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

public class NoiseCosineGradientCalcDeep extends GradientCalc {

  public NoiseCosineGradientCalcDeep(Batch _batch) {
    super(_batch);
  }

  // f for the parameter data 
  public void testStats (Batch _batch) {
//    System.out.printf("calling getValue()\n");
    double err = 0.0;
    double batchError = 0.0;
    int nDP = 1;
    if(System.getProperty("use_cuda").equals("true")) {

      this.model.copyHtoD();

      DMatrix A = this.model.fProp(_batch.data());
      DMatrix B = this.model.fProp(_batch.pos());
      DMatrix N = this.model.fProp(_batch.neg());


      DMatrix ARep = this.prepareSumMatrix(A, _batch.dataWordsArray());
      DMatrix BRep = this.prepareSumMatrix(B, _batch.posWordsArray());
      DMatrix NRep = this.prepareSumMatrix(N, _batch.negWordsArray());

      ARep.copyHtoD();
      BRep.copyHtoD();
      NRep.copyHtoD();

      DMatrix ARepNorm = ARep.vectorNorm();
      DMatrix BRepNorm = BRep.vectorNorm();
      DMatrix NRepNorm = NRep.vectorNorm();


      double posPart = (double) ARepNorm.dotRows(BRepNorm).sumRows().get(0);
      double negPart = (double) ARepNorm.dotRows(NRepNorm).sumRows().get(0);


      ARep.close();
      BRep.close();
      NRep.close();
      
      // f = cos(A, B) - cos(A, N)
      batchError = (posPart - negPart)/(_batch.nSamples()*_batch.size());

      this.testLoss = batchError;
      this.testMRR = Metric.mrr(ARepNorm.mmul(false, true, BRepNorm));
      this.model.clearDevice();
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

      this.model.copyHtoD();
        DMatrix A = this.model.fProp(this.batch.data());
        DMatrix B = this.model.fProp(this.batch.pos());
        DMatrix N = this.model.fProp(this.batch.neg());

/*        A.copyHtoD();
        B.copyHtoD();
        N.copyHtoD();*/


        DMatrix ARep = this.prepareSumMatrix(A, this.batch.dataWordsArray());
        DMatrix BRep = this.prepareSumMatrix(B, this.batch.posWordsArray());
        DMatrix NRep = this.prepareSumMatrix(N, this.batch.negWordsArray());

        ARep.copyHtoD();
        BRep.copyHtoD();
        NRep.copyHtoD();

        DMatrix ARepNorm = ARep.vectorNorm();
        DMatrix BRepNorm = BRep.vectorNorm();
        DMatrix NRepNorm = NRep.vectorNorm();


        double posPart = (double) ARepNorm.dotRows(BRepNorm).sumRows().get(0);
        double negPart = (double) ARepNorm.dotRows(NRepNorm).sumRows().get(0);

//        System.out.printf("Pos part = %.10f\tNeg Part = %.10f\n", posPart, negPart);


        // TODO: POW is not accurate. Not enough to have accurate gradient computation.
        
//        double negPart = 0.5*(double)(((ARep.sub(BRep)).powi(2.0)).sumColumns().sumRows().get(0));
//        double posPart = 0.5*(double)(((ARep.sub(NRep)).powi(2.0)).sumColumns().sumRows().get(0));

//        double negPart = 0.5*(double)(((ARep.sub(BRep)).mul(ARep.sub(BRep))).sumColumns().sumRows().get(0));
//        double posPart = 0.5*(double)(((ARep.sub(NRep)).mul(ARep.sub(NRep))).sumColumns().sumRows().get(0));

        ARep.close();
        BRep.close();
        NRep.close();
        
        // f = cos(A, B) - cos(A, N)
        batchError = (posPart - negPart)/(this.batch.nSamples()*this.batch.size());

//        System.out.printf("Error = %.10f\n", batchError);

        this.model.clearDevice();
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
      this.model.copyHtoD();
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

        ARep.copyHtoD();
        BRep.copyHtoD();
        NRep.copyHtoD();

        DMatrix aPos = ARep.dotRows(BRep);
        DMatrix b = ARep.rowNorms().inv();
        DMatrix cPos = BRep.rowNorms().inv();

        DMatrix aNeg = ARep.dotRows(NRep);
        DMatrix cNeg = NRep.rowNorms().inv();

        aPos.copyHtoD();
        b.copyHtoD();
        cPos.copyHtoD();
        aNeg.copyHtoD();
        cNeg.copyHtoD();

        // calcuating offsets at once.
        DMatrix b3 = b.mul(b).mul(b);
//        DMatrix b3 = b.pow(3.0);
        DMatrix cPos3 = cPos.mul(cPos).mul(cPos);
        DMatrix cNeg3 = cNeg.mul(cNeg).mul(cNeg);
        
        DMatrix posQ = b.mul(cPos);
        DMatrix posSQ = aPos.mul(cPos).mul(b3);
        DMatrix posSD = aPos.mul(b).mul(cPos3);

        DMatrix negQ = b.mul(cNeg);
        DMatrix negSQ = aNeg.mul(cNeg).mul(b3);
        DMatrix negSD = aNeg.mul(b).mul(cNeg3);


        aPos.close();
        b.close();
        cPos.close();
        aNeg.close();
        cNeg.close();

        posQ.copyHtoD();
        posSQ.copyHtoD();
        posSD.copyHtoD();
        negQ.copyHtoD();
        negSQ.copyHtoD();
        negSD.copyHtoD();
        
        DMatrix dAB = BRep.mulRows(posQ).sub(ARep.mulRows(posSQ));
        DMatrix dBA = ARep.mulRows(posQ).sub(BRep.mulRows(posSD));
        
        DMatrix dAN = NRep.mulRows(negQ).sub(ARep.mulRows(negSQ));
        DMatrix dNA = ARep.mulRows(negQ).sub(NRep.mulRows(negSD));

        posQ.close();
        posSQ.close();
        posSD.close();
        negQ.close();
        negSQ.close();
        negSD.close();

        dAB.copyHtoD();
        dBA.copyHtoD();
        dAN.copyHtoD();
        dNA.copyHtoD();

        DMatrix AB = this.prepareErrorMatrix(dAB, this.batch.dataWordsArray(), A.rows());
        DMatrix BA = this.prepareErrorMatrix(dBA, this.batch.posWordsArray(), B.rows());
        DMatrix AN = this.prepareErrorMatrix(dAN, this.batch.dataWordsArray(), A.rows());
        DMatrix NA = this.prepareErrorMatrix(dNA, this.batch.negWordsArray(), N.rows());

        dAB.close();
        dBA.close();
        dAN.close();
        dNA.close();

        AB.copyHtoD();
        BA.copyHtoD();
        AN.copyHtoD();
        NA.copyHtoD();

        // df/dA = (A, B) - (A, N)
        DMatrix tempGrad = this.model.bProp(this.batch.data(), AB);
        grads.addi(tempGrad);
        tempGrad.close();

        tempGrad = this.model.bProp(this.batch.data(), AN);
        grads.subi(tempGrad);
        tempGrad.close();

        // df/dB = (B, A)
        tempGrad = this.model.bProp(this.batch.pos(), BA);
        grads.addi(tempGrad);
        tempGrad.close();

        // df/dN = - (N, A)
        tempGrad = this.model.bProp(this.batch.neg(), NA);
        grads.subi(tempGrad);
        tempGrad.close();

        grads.muli(1.0/(nSamples*this.batch.size()));
        
        AB.close();
        BA.close();
        AN.close();
        NA.close();
        
        ARep.close();
        BRep.close();
        NRep.close();
        
        A.close();
        B.close();
        N.close();

        grads.copyDtoH();
        grads.close();
        this.model.clearDevice();
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

