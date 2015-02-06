package common;


import java.util.Map;
import java.util.HashMap;
import java.util.Iterator;
import java.io.PrintWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;

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

  public Map<String, Integer> str2id() {
    return this.str2id;
  }
  
  public boolean contains(String token) {
    return this.str2id.containsKey(token);
  }
 
  public int getSize() {
    return this.dictSize;
  }
  public void addWord(String t) {
    if(!this.str2id.containsKey(t) && t.trim().length()>0) {
      this.str2id.put(t,this.dictSize);
      this.id2str.put(this.dictSize,t);
      this.dictSize++;
    }
  }

  public void addWord(int id, String t) {
    assert (!this.str2id.containsKey(t)): "Term already present";
    assert (!this.id2str.containsKey(id)): "Term Id already present";
    this.str2id.put(t, id);
    this.id2str.put(id, t);
    this.dictSize++;
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

  public DMatrix getRepresentation(Sentence sent) {
    DMatrix mat = DMath.createZerosMatrix(sent.getSize(), dictSize);
    Iterator<Integer> sentIt = sent.words.iterator();
    int i = 0;
    //TODO: This will add all the words including OOV as Zero vector, do something to consider active vocab
    while(sentIt.hasNext()) {
      mat.fillRow(i, this.getRepresentation(sentIt.next()));
      i++;
    }
    return mat;
  }

  public DMatrix getRepresentation(Sentence[] sents) {
    int row = 0;
    for(int i=0; i<sents.length; i++)
      row+= sents[i].getSize();
    DMatrix mat = DMath.createMatrix(row, dictSize);
    row=0; 
    for(int i=0; i<sents.length; i++) {
      DMatrix sentMat = this.getRepresentation(sents[i]);
      mat.fillMatrix(row, sentMat);
      row+=sents[i].getSize();
    }
    return mat;
  }

  public int getId(String t) {
    return this.str2id.containsKey(t)?this.str2id.get(t):-1;
  }

  public String getTerm(int id) {
    return this.id2str.containsKey(id)?this.id2str.get(id):null;
  }

  public void save(String file) throws IOException {
    PrintWriter p = new PrintWriter(file, "UTF-8");
    for(int i=0; i<this.dictSize; i++)
      p.printf("%d\t%s\n", i, this.id2str.get(i));
    p.close();
  }

  public void load(String file) throws IOException {
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
    String line = "";
    while((line = br.readLine())!=null) {
//      System.out.printf("%s%n", line);
      String[] cols = line.split("\t");
      this.addWord(Integer.parseInt(cols[0].trim()), cols[1].trim()); 
    }
    br.close();
  }

  public void mergeDict(Dictionary dict2) {
    for(int i=0; i<dict2.getSize(); i++) {
      this.addWord(dict2.getTerm(i));
    }
    System.out.printf("Dictionary added. Total terms after addition = %d\n", this.getSize());
  }
}
