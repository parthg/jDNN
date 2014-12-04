package data;

import java.util.Map;

import es.upv.nlel.corpus.NEWSDocType;
import es.upv.nlel.parser.FIREParserInterface;

public class AdHoc extends Channel {
  Map<Integer, String> docIndex;
  String dataPath;
  FIREParserInterface parser;
  String ext;
  
  public AdHoc(String path_to_data, String _ext) {
    this.dataPath = path_to_data;
    this.ext = _ext;
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
