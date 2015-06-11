package common;

import java.util.List;
import java.util.ArrayList;


public class Sentence {
  int id;
  String label;
  public List<Integer> words;
  int sentLength;

  public Sentence() {
    this.words = new ArrayList<Integer>();
    this.sentLength = 0;
  }

  public Sentence copy() {
    Sentence s = new Sentence();
    s.words = new ArrayList<Integer>();
    s.words.addAll(this.words);
    s.id = this.id;
    s.label = this.label;
    s.sentLength = this.sentLength;
    return s;
  }
  // returns the ith word id
  public int get(int i) {
    assert (i<this.getSize());
    return this.words.get(i);
  }

  public void setId(int _id) {
    this.id = _id;
  }

  public int id() {
    return this.id;
  }

  public void setLabel(String _label) {
    this.label = _label;
  }

  public String label() {
    return this.label;
  }

  public void addWord(Integer t) {
    words.add(t);
    this.sentLength++;
  }

  public void addSent(Sentence s2) {
    for(int i=0; i<s2.getSize(); i++) {
      this.addWord(s2.get(i));
    }
  }

  public int getSize() {
    return this.words.size();
  }

  public String toString() {
    StringBuffer sb = new StringBuffer();
    for(int word: words)
      sb.append(word+ " ");

    return sb.toString();
  }
}
