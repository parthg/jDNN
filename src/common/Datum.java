package common;

//import common.Sentence;
import java.util.List;

public class Datum {
  int id; 
  int label;
  Sentence data;
  Sentence posData;
  List<Sentence> negData;
  
  public int getNegSampleSize() {
    return this.negData.size();
  }

  public Sentence getData() {
    return this.data;
  }

  public Sentence getPos() {
    return this.posData;
  }

  public List<Sentence> getNeg() {
    return this.negData;
  }

  public Datum(int _id, Sentence _data, Sentence _posData, List<Sentence> _negData) {
    this.id = _id;
    this.label = _id; // in absence of explicit labels, ids can be used as labels. e.g. retrieval tasks
    this.data = _data;
    this.posData = _posData;
    this.negData = _negData;
  }
}
