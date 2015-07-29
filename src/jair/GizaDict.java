package jair;

import java.util.Map;
import java.util.HashMap;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.IOException;

public class GizaDict {
  Map<String, String> dict;
  Map<String, Double> scores;

  public Map<String, String> dict() {
    return this.dict();
  }

  public Map<String, Double> scores() {
    return this.scores();
  }

  public boolean contains(String term) {
    return this.dict.containsKey(term);
  }

  public String get(String term) {
    if(this.dict.containsKey(term))
      return this.dict.get(term);
    else {
      System.err.printf("Term %s is OOV for this GizaDict. Please check contains() before calling get() \n", term);
      return null;
    }
  }

  public void load(String e2f) throws IOException {
    this.dict = new HashMap<String, String>();
    this.scores = new HashMap<String, Double>();
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(e2f), "UTF-8"));
    String line = "";
    while((line = br.readLine())!=null) {
      String[] cols = line.split(" ");
      String s = cols[0].trim();
      String t = cols[1].trim();
      double val = Double.parseDouble(cols[2].trim());
      if(this.dict.containsKey(s)) {
        if(this.scores.get(s)<val) {
          this.dict.put(s, t);
          this.scores.put(s, val);
        }
      }
      else{
        this.dict.put(s, t);
        this.scores.put(s, val);
      }
    }
    br.close();
  }
  
  public void load(String e2f, String f2e) throws IOException {
    this.dict = new HashMap<String, String>();
    this.scores = new HashMap<String, Double>();
/*    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(e2f), "UTF-8"));
    String line = "";
    while((line = br.readLine())!=null) {
      String[] cols = line.split(" ");
      String s = cols[0].trim();
      String t = cols[1].trim();
      double val = Double.parseDouble(cols[2].trim());
      if(this.dict.containsKey(s)) {
        if(this.scores.get(s)<val) {
          this.dict.put(s, t);
          this.scores.put(s, val);
        }
      }
      else{
        this.dict.put(s, t);
        this.scores.put(s, val);
      }
    }
    br.close();*/


    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f2e), "UTF-8"));
    String line = "";
    while((line = br.readLine())!=null) {
      String[] cols = line.split(" ");
      String t = cols[0].trim();
      String s = cols[1].trim();
      double val = Double.parseDouble(cols[2].trim());
      if(this.dict.containsKey(s)) {
        if(this.scores.get(s)<val) {
          this.dict.put(s, t);
          this.scores.put(s, val);
        }
      }
      else{
        this.dict.put(s, t);
        this.scores.put(s, val);
      }
    }
    br.close();
  }
}
