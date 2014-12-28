package common;


import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;

import org.jblas.DoubleMatrix;

public class Dictionary {
  Map<String, Integer> str2id;
  Map<Integer, String> id2str;
  int dictSize;

  public Dictionary() {
    this.str2id = new HashMap<String, Integer>();
    this.id2str = new HashMap<Integer, String>();
    this.dictSize = 0;
  }

  public void print() {
    Iterator<Integer> it = this.id2str.keySet().iterator();
    while(it.hasNext()) {
      int termId = it.next();
      System.out.printf("%d\t%s\n", termId, this.id2str.get(termId));
    }
  }
 
  public int getSize() {
    return this.dictSize;
  }
  public void addWord(String t) {
    if(!this.str2id.containsKey(t)) {
      this.str2id.put(t,this.dictSize);
      this.id2str.put(this.dictSize,t);
      this.dictSize++;
    }
  }
  public DoubleMatrix getRepresentation(String t) {
    DoubleMatrix vec = DoubleMatrix.zeros(1,dictSize);
    if(str2id.containsKey(t))
      vec.put(0,this.str2id.get(t),1.0);
    return vec;
  }

  public DoubleMatrix getRepresentation(int t) {
    DoubleMatrix vec = DoubleMatrix.zeros(1,dictSize);
    if(id2str.containsKey(t))
      vec.put(0,t, 1.0);
    return vec;
  }

  public int getId(String t) {
    return this.str2id.containsKey(t)?this.str2id.get(t):-1;
  }
  
}
