package optim;

import models.Model;
import common.Sentence;
import common.Datum;
import org.jblas.DoubleMatrix;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import java.util.List;

public class MonoNoiseCost {
  Model model;
  double cost;
  DoubleMatrix grads;
  Lock lock;

  public MonoNoiseCost(Model _model) {
    this.model = _model;
    this.lock = new ReentrantLock();
    this.cost = 0.0;
    this.grads = DoubleMatrix.zeros(1, this.model.getThetaSize());
  }

  public double getCost() {
    return this.cost;
  }

  public DoubleMatrix getGrads() {
    return this.grads;
  }
  public void computeCost(Datum d) {
    DoubleMatrix s1_root = this.model.fProp(d.getData());
    DoubleMatrix s2_root = this.model.fProp(d.getPos());
    int nSamples = d.getNegSampleSize();
    List<Sentence> neg = d.getNeg();
    double unitError = 0.0;
    for(int i=0; i<nSamples; i++) {
      DoubleMatrix s3_root = this.model.fProp(neg.get(i));

      unitError += 0.5*Math.pow(s1_root.distance2(s2_root),2)-0.5*Math.pow(s1_root.distance2(s3_root),2);
    }
    unitError = unitError/nSamples;
    lock.lock(); 
    {
      this.cost += unitError;
    }
    lock.unlock();
  }

  public void computeGrad(Datum d) {
    int nSamples = (d.getNegSampleSize()>0)?d.getNegSampleSize():1;
    List<Sentence> neg = d.getNeg();
    
    DoubleMatrix unitGrads = DoubleMatrix.zeros(1, this.model.getThetaSize());

    for(int i=0; i<nSamples; i++) {
      // df/dA = (A-B) - (A-N)
      unitGrads.addi(this.model.bProp(d.getData(), d.getPos()));
      unitGrads.subi(this.model.bProp(d.getData(), neg.get(i)));

      // df/dB = (B-A)
      unitGrads.addi(this.model.bProp(d.getPos(), d.getData()));

      // df/dN = - (N-A)
      unitGrads.subi(this.model.bProp(neg.get(i), d.getData()));
    }
    unitGrads.muli(1.0/nSamples);
    lock.lock();
    {
      this.grads.addi(unitGrads);
    }
    lock.unlock();
  }
}
