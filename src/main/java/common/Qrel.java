package common;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Map;
import java.util.HashMap;

import util.StringUtils;
public class Qrel {
  Map<Integer, Map<String, Integer>> qrel;
  public Qrel(String qrelFile) throws IOException {
    this.qrel = new HashMap<Integer, Map<String, Integer>>();
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(qrelFile), "UTF-8"));
    String line = "";
    while((line = br.readLine())!=null) {
      String[] cols = line.split(" ");
      int qid = Integer.parseInt(cols[0].trim());
      String did = StringUtils.removeExt(StringUtils.fileName(cols[2].trim()));
      int rel = Integer.parseInt(cols[3].trim());
      if(rel>0) {
        if(!this.qrel.containsKey(qid)) {
          Map<String, Integer> inner = new HashMap<String, Integer>();
          inner.put(did, rel);
          this.qrel.put(qid, inner);
        }
        else {
          Map<String, Integer> inner = this.qrel.get(qid);
          inner.put(did, rel);
          this.qrel.put(qid, inner);
        }
      }
    }
    br.close();
  }
}
