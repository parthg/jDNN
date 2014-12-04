package data;

import java.util.Map;

import es.upv.nlel.corpus.NEWSDocType;
import es.upv.nlel.parser.FIREParserInterface;

public class AdHoc extends Channel {

  public AdHoc() {
  }
  
  public void setParser(NEWSDocType type) {
//  	not required
  }
  
  public Map<String, Integer> getTokensFreq() {
    return this.tokenFreq;
  }
  
  public Map<Integer, Map<Integer, Integer>> getMatrix() {
  	return null;
  }
}
