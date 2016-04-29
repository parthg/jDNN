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

public class CLNoiseCosineGradientCalcL2 extends GradientCalc {

  public CLNoiseCosineGradientCalcL2(Batch _batch) {
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
      DMatrix B = _batch.pos();
      DMatrix N = _batch.neg();


      DMatrix ARep = this.prepareSumMatrix(A, _batch.dataWordsArray());

      ARep.copyHtoD();

      DMatrix ARepNorm = ARep.vectorNorm();
      DMatrix BRepNorm = B.vectorNorm();
      DMatrix NRepNorm = N.vectorNorm();

      double posPart = (double) ARepNorm.dotRows(BRepNorm).sumRows().get(0);
      double negPart = (double) ARepNorm.dotRows(NRepNorm).sumRows().get(0);
      double regPart = (this.model.lambda()*0.5)*this.model.weightSquaredSum();

      ARep.close();
      
      // f = cos(A, B) - cos(A, N) - \lambda/2 \sum w^2
      batchError = (posPart - negPart - regPart)/(_batch.nSamples()*_batch.size());

      this.testLoss = batchError;
      this.testMRR = Metric.mrr(ARepNorm.mmul(false, true, BRepNorm));
      this.model.clearDevice();
    }
    // make it parallel
    else {
      throw new UnsupportedOperationException("TODO");
/*      final MonoNoiseCost costComputer = new MonoNoiseCost(this.model);
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
      this.testLoss = err/_batch.size();*/
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
        DMatrix B = this.batch.pos();
        DMatrix N = this.batch.neg();

/*        A.copyHtoD();
        B.copyHtoD();
        N.copyHtoD();*/

        DMatrix ARep = this.prepareSumMatrix(A, this.batch.dataWordsArray());
        ARep.copyHtoD();

        DMatrix ARepNorm = ARep.vectorNorm();
        DMatrix BRepNorm = B.vectorNorm();
        DMatrix NRepNorm = N.vectorNorm();

        double posPart = (double) ARepNorm.dotRows(BRepNorm).sumRows().get(0);
        double negPart = (double) ARepNorm.dotRows(NRepNorm).sumRows().get(0);
        double regPart = (this.model.lambda()*0.5)*this.model.weightSquaredSum();

        ARep.close();
        
        // f = cos(A, B) - cos(A, N) - \lambda/2 \sum w^2
        batchError = (posPart - negPart - regPart)/(this.batch.nSamples()*this.batch.size());

//        System.out.printf("Error = %.10f\n", batchError);

        this.model.clearDevice();
      return batchError;
    }
    // make it parallel
    else {
      throw new UnsupportedOperationException("TODO");
/*      final MonoNoiseCost costComputer = new MonoNoiseCost(this.model);
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
      return err/this.batch.size();*/
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
        DMatrix B = this.batch.pos();
        DMatrix N = this.batch.neg();

        A.copyHtoD();

        DMatrix ARep = this.prepareSumMatrix(A, this.batch.dataWordsArray());

        ARep.copyHtoD();

        DMatrix aPos = ARep.dotRows(B);
        DMatrix b = ARep.rowNorms().inv();
        DMatrix cPos = B.rowNorms().inv();

        DMatrix aNeg = ARep.dotRows(N);
        DMatrix cNeg = N.rowNorms().inv();

        aPos.copyHtoD();
        b.copyHtoD();
        cPos.copyHtoD();
        aNeg.copyHtoD();
        cNeg.copyHtoD();

        // calcuating offsets at once.
        DMatrix b3 = b.mul(b).mul(b);
//        DMatrix b3 = b.pow(3.0);
        
        DMatrix posQ = b.mul(cPos);
        DMatrix posSQ = aPos.mul(cPos).mul(b3);

        DMatrix negQ = b.mul(cNeg);
        DMatrix negSQ = aNeg.mul(cNeg).mul(b3);


        aPos.close();
        b.close();
        cPos.close();
        aNeg.close();
        cNeg.close();

        posQ.copyHtoD();
        posSQ.copyHtoD();
        negQ.copyHtoD();
        negSQ.copyHtoD();
        
        DMatrix dAB = B.mulRows(posQ).sub(ARep.mulRows(posSQ));
        
        DMatrix dAN = N.mulRows(negQ).sub(ARep.mulRows(negSQ));

        posQ.close();
        posSQ.close();
        negQ.close();
        negSQ.close();

        dAB.copyHtoD();
        dAN.copyHtoD();

        DMatrix AB = this.prepareErrorMatrix(dAB, this.batch.dataWordsArray(), A.rows());
        DMatrix AN = this.prepareErrorMatrix(dAN, this.batch.dataWordsArray(), A.rows());

        dAB.close();
        dAN.close();

        AB.copyHtoD();
        AN.copyHtoD();

        // df/dA = (A, B) - (A, N)
        DMatrix tempGrad = this.model.bProp(this.batch.data(), A, AB);
        grads.addi(tempGrad);
        tempGrad.close();

        tempGrad = this.model.bProp(this.batch.data(), A, AN);
        grads.subi(tempGrad);
        tempGrad.close();


        grads.muli(1.0/(nSamples*this.batch.size()));
        
        // Regularization
        DMatrix regPart = DMath.createMatrix(grads.rows(), grads.columns(), this.model.getWeightOnlyParameters());
        regPart.muli(this.model.lambda()/(nSamples*this.batch.size()));
        grads.subi(regPart);
        
        AB.close();
        AN.close();
        
        ARep.close();
        
        A.close();

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

