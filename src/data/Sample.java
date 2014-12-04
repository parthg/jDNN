package data;

import java.util.Set;
import java.util.HashSet;
import java.io.Serializable;

public class Sample implements Serializable {
  static final long serialVersionUID = 42L; 
  Set<String> pos;
  Set<String> neg;
  
  public Sample() {
    this.pos = new HashSet<String>();
    this.neg = new HashSet<String>();
  }

  public void addPos(String txt) {
    this.pos.add(txt);
  }

  public void addNeg(String txt) {
    this.neg.add(txt);
  }

  public Set<String> getPos() {
    return this.pos;
  }

  public Set<String> getNeg() {
    return this.neg;
  }

  public int getPosSize() {
    return this.pos.size();
  }
  
  public int getNegSize() {
    return this.neg.size();
  }
}
