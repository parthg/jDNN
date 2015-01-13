package optim;

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

        int nSamples = d.getNegSampleSize();
        nDP = nSamples;
        List<Sentence> neg = d.getNeg();
  
        DMatrix dataIn = this.model.dict().getRepresentation(d.getData());
        DMatrix dataPos = this.model.dict().getRepresentation(d.getPos());

        DMatrix s1_root = this.model.getRepresentation(dataIn);
        DMatrix s2_root = this.model.getRepresentation(dataPos);

        double unitError = 0.0;
        for(int i=0; i<nSamples; i++) {

          DMatrix dataNeg = this.model.dict().getRepresentation(neg.get(i));
          DMatrix s3_root = this.model.getRepresentation(dataNeg);

          // f = 1/2(A-N)^2 - 1/2(A-B)^2
          unitError += -(0.5*s1_root.squaredDistance(s2_root))+(0.5*s1_root.squaredDistance(s3_root));
        }
        err+=unitError;
      }
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
      int nSamples = 1;
      if(System.getProperty("use_cuda").equals("true")) {

//        System.out.printf("Total DP = %d\n", this.data.size());

        Sentence[] dataSents = new Sentence[this.data.size()];
        Sentence[] posSents = new Sentence[this.data.size()];
        Sentence[] negSents = new Sentence[nSamples*this.data.size()];

        int[] dataWords = new int[this.data.size()];
        int[] posWords = new int[this.data.size()];
        int[] negWords = new int[nSamples*this.data.size()];

        //TODO: Probably its a good idea to store data and pos also nSamples times for the easy calculation
        for(int dp=0; dp<this.data.size(); dp++) {
          dataSents[dp] = this.data.get(dp).getData();
          dataWords[dp] = this.data.get(dp).getData().getSize();

          posSents[dp] = this.data.get(dp).getPos();
          posWords[dp] = this.data.get(dp).getPos().getSize();

          List<Sentence> neg = this.data.get(dp).getNeg();
          for(int i=0; i<nSamples; i++) {
            negSents[dp*nSamples+i] = neg.get(i);
            negWords[dp*nSamples+i] = neg.get(i).getSize();
          }
          
        }
        
        DMatrix dataIn = this.model.dict().getRepresentation(dataSents);
        DMatrix dataPos = this.model.dict().getRepresentation(posSents);
        DMatrix dataNeg = this.model.dict().getRepresentation(negSents);

        dataIn.copyHtoD();
        dataPos.copyHtoD();
        dataNeg.copyHtoD();

        
        DMatrix A = this.model.fProp(dataIn);
        DMatrix B = this.model.fProp(dataPos);
        DMatrix N = this.model.fProp(dataNeg);

        A.copyHtoD();
        B.copyHtoD();
        N.copyHtoD();


        DMatrix ARep = this.prepareSumMatrix(A, dataWords);
        DMatrix BRep = this.prepareSumMatrix(B, posWords);
        DMatrix NRep = this.prepareSumMatrix(N, negWords);

        DMatrix AB = this.prepareErrorMatrix(ARep.sub(BRep), dataWords, A.rows());
        DMatrix BA = this.prepareErrorMatrix(BRep.sub(ARep), posWords, B.rows());
        DMatrix AN = this.prepareErrorMatrix(ARep.sub(NRep), dataWords, A.rows());
        DMatrix NA = this.prepareErrorMatrix(NRep.sub(ARep), negWords, N.rows());

        AB.copyHtoD();
        BA.copyHtoD();
        AN.copyHtoD();
        NA.copyHtoD();

        // df/dA = (A-N) - (A-B)
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

        grads.muli(1.0/(nSamples*this.data.size()));

        dataIn.close();
        dataPos.close();
        dataNeg.close();

        A.close();
        B.close();
        N.close();

        AB.close();
        BA.close();
        AN.close();
        NA.close();
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

