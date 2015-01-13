package common;

import java.lang. AutoCloseable;
import math.DMatrix;
import common.Datum;
import java.util.List;

public class Batch implements AutoCloseable {
  List<Datum> listData;
  int nSamples;
  int batchSize;
  DMatrix data;
  DMatrix pos;
  DMatrix neg;

  int[] dataWords;
  int[] posWords;
  int[] negWords;

  public Batch(List<Datum> instances, int _nSamples, Dictionary dict) {

    this.listData = instances;
    this.nSamples = _nSamples;
    this.batchSize = instances.size();

    // BTW, below is needed only for the GPU thing, so make it conditional

    Sentence[] dataSents = new Sentence[instances.size()];
    Sentence[] posSents = new Sentence[instances.size()];
    Sentence[] negSents = new Sentence[_nSamples*instances.size()];

    this.dataWords = new int[this.batchSize];
    this.posWords = new int[this.batchSize];
    this.negWords = new int[nSamples*this.batchSize];

    //TODO: Probably its a good idea to store data and pos also nSamples times for the easy calculation
    for(int dp=0; dp<this.batchSize; dp++) {
      dataSents[dp] = instances.get(dp).getData();
      this.dataWords[dp] = instances.get(dp).getData().getSize();

      posSents[dp] = instances.get(dp).getPos();
      this.posWords[dp] = instances.get(dp).getPos().getSize();

      List<Sentence> neg = instances.get(dp).getNeg();
      for(int i=0; i<this.nSamples; i++) {
        negSents[dp*this.nSamples+i] = neg.get(i);
        negWords[dp*this.nSamples+i] = neg.get(i).getSize();
      }
    }
    
    this.data = dict.getRepresentation(dataSents);
    this.pos = dict.getRepresentation(posSents);
    this.neg = dict.getRepresentation(negSents);
  }

  public List<Datum> listData() {
    return this.listData;
  }
  public DMatrix data() {
    return data;
  }

  public DMatrix pos() {
    return pos;
  }

  public DMatrix neg() {
    return neg;
  }

  public int[] dataWordsArray() {
    return dataWords;
  }

  public int[] posWordsArray() {
    return posWords;
  }

  public int[] negWordsArray() {
    return negWords;
  }

  public int nSamples() {
    return nSamples;
  }

  public int size() {
    return batchSize;
  }

  public void copyHtoD() {
    this.data.copyHtoD();
    this.pos.copyHtoD();
    this.neg.copyHtoD();
  }

  public void close() {
    this.data.close();
    this.pos.close();
    this.neg.close();
  }
}
