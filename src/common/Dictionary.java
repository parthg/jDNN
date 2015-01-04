package common;


import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;
import java.io.PrintWriter;
import java.io.IOException;

//import org.jblas.DoubleMatrix;
import math.DMath;
import math.DMatrix;

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
  public DMatrix getRepresentation(String t) {
    DMatrix vec = DMath.createZerosMatrix(1,dictSize);
    if(str2id.containsKey(t))
      vec.put(0,this.str2id.get(t),1.0);
    return vec;
  }

  public DMatrix getRepresentation(int t) {
    DMatrix vec = DMath.createZerosMatrix(1,dictSize);
    if(id2str.containsKey(t))
      vec.put(0,t, 1.0);
    return vec;
  }

  public int getId(String t) {
    return this.str2id.containsKey(t)?this.str2id.get(t):-1;
  }

  public void save(String file) throws IOException {
    PrintWriter p = new PrintWriter(file, "UTF-8");
    for(int i=0; i<this.dictSize; i++)
      p.printf("%d\t%s\n", i, this.id2str.get(i));
    p.close();
  }
  
}
