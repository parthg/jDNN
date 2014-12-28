package common;

import java.util.List;
import java.util.ArrayList;


public class Sentence {
  int id;
  public List<Integer> words;
  int sentLength;

  Sentence() {
    this.words = new ArrayList<Integer>();
    this.sentLength = 0;
  }

  public void setId(int _id) {
    this.id = _id;
  }

  public void addWord(Integer t) {
    words.add(t);
    this.sentLength++;
  }
}
