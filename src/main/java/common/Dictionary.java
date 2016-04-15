package common;

import java.util.List;
import java.util.ArrayList;
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
  Map<Integer, Double> idf;
  int dictSize;

  public Dictionary() {
    this.str2id = new HashMap<String, Integer>();
    this.id2str = new HashMap<Integer, String>();
    this.idf = new HashMap<Integer, Double>();
    this.dictSize = 0;
  }

  public void setIdf(String s, double val) {
    assert (this.str2id.containsKey(s)): System.out.printf("Term: %s is not present in the dictionary", s);
/*    if(val == 0.0)
      val = 1.0;*/
    this.idf.put(this.getId(s), val);
  }

  public double getIdf(String s) {
    assert (this.str2id.containsKey(s)): System.out.printf("Term: %s is not present in the dictionary", s);
    return this.idf.get(this.getId(s));
  }

  public double getIdf(int id) {
    assert (this.id2str.containsKey(id)): System.out.printf("TermId: %d is not present in the dictionary", id);
    return this.idf.get(id);
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
    else
      throw new IllegalArgumentException("Term does not exist in the Dictionary: " + t);
    return vec;
  }

  public DMatrix getRepresentation(int t) {
    DMatrix vec = DMath.createZerosMatrix(1,dictSize);
    if(id2str.containsKey(t))
      vec.put(0,t, 1.0);
    else
      throw new IllegalArgumentException("Term id does not exist in the Dictionary: " + t);
    return vec;
  }

  /** Converts the input sentence into a Bag-of-Words vector representation.
   *
   * @param sent  input text as Sentence
   * @return  The BoW representation as DMatrix (vector)
   */
  public DMatrix getBoWRepresentation(Sentence sent) {
//    throw new UnsupportedOperationException("UNIMPLEMENTED");
    DMatrix vec = DMath.createZerosMatrix(1, dictSize);
    if(sent.getSize()==0) {
      throw new IllegalArgumentException("Sentence:\"" + sent.toString() + "\" does not have any Dictionary term.");
    }
    Iterator<Integer> sentIt = sent.words.iterator();
    while(sentIt.hasNext()) {
      int id = sentIt.next();
      if(this.id2str.containsKey(id)) {
        vec.put(0, id, vec.get(0, id)+1.0);
      }
    }
    return vec;
  }

  public DMatrix getAddRepresentation(Sentence sent) {
    DMatrix mat = DMath.createZerosMatrix(sent.getSize(), dictSize);
    if(sent.getSize()==0) {
      throw new IllegalArgumentException("None of the sentence terms appear in the Dictionary.");
    }
    Iterator<Integer> sentIt = sent.words.iterator();
    int i = 0;
    while(sentIt.hasNext()) {
      mat.fillRow(i, this.getRepresentation(sentIt.next()));
      i++;
    }
    return mat;    
  }
  
  public DMatrix getRepresentation(Sentence sent) {
    if(System.getProperty("representation") ==  null)
      throw new IllegalArgumentException("Please set \"representation\" System property: [bow|add].");
    else if(System.getProperty("representation").equals("bow"))
      return this.getBoWRepresentation(sent);
    else if(System.getProperty("representation").equals("add"))
      return this.getAddRepresentation(sent);
    else
      throw new IllegalArgumentException("Please set proper \"representation\" System property [bow|add]. Invalid value : " + System.getProperty("representation"));
  }

  public DMatrix getBoWRepresentation(Sentence[] sents) {
//    throw new UnsupportedOperationException("UNIMPLEMENTED");
    int row = 0;
    for(int i=0; i<sents.length; i++)
      row+= (sents[i].getSize()>0)?1:0;
    if(row==0) {
      throw new IllegalArgumentException("None of the sentences appear in the Dictionary.");
    }
    DMatrix mat = DMath.createMatrix(row, dictSize);
    row=0; 
    for(int i=0; i<sents.length; i++) {
      if(sents[i].getSize()>0) {
        DMatrix sentMat = this.getBoWRepresentation(sents[i]);
        mat.fillMatrix(row, sentMat);
        row++;
      }
    }
    return mat;
  }

  public DMatrix getAddRepresentation(Sentence[] sents) {
    int row = 0;
    for(int i=0; i<sents.length; i++)
      row+= sents[i].getSize();
    if(row==0) {
      throw new IllegalArgumentException("None of the sentences appear in the Dictionary.");
    }
    DMatrix mat = DMath.createMatrix(row, dictSize);
    row=0; 
    for(int i=0; i<sents.length; i++) {
      if(sents[i].getSize()>0) {
        DMatrix sentMat = this.getAddRepresentation(sents[i]);
        mat.fillMatrix(row, sentMat);
        row+=sents[i].getSize();
      }
    }
    return mat;
  }

  public DMatrix getRepresentation(Sentence[] sents) {
    if(System.getProperty("representation").equals("bow"))
      return this.getBoWRepresentation(sents);
    else if(System.getProperty("representation").equals("add"))
      return this.getAddRepresentation(sents);
    else
      throw new IllegalArgumentException("Please set proper \"representation\" System property: [bow|add].");
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

  public List<Integer> getTermIds(List<String> tokens) {
    List<Integer> tids = new ArrayList<Integer>();
    for(String t: tokens) {
      if(this.contains(t.trim()))
        tids.add(this.getId(t.trim()));
    }
    return tids;
  }
}
