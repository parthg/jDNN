package data;

import java.util.List;
import java.util.Map;

import es.upv.nlel.corpus.NEWSDocType;
import es.upv.nlel.utils.Language;

/** Channel Abstract class is responsible for input channel of data. Input channel can be a directory for which all the files with a particular extension or a per-line sentence file.
 * To use it, fist make an instance of it with the implementing class (DocCollection or SentFile), setup and setParser.
 *
 * @author Parth Gupta
 */
public abstract class Channel {
	public abstract void setParser(NEWSDocType type);
	
  public abstract void setup(TokenType tokenType, Language _lang, String path_to_terrier, List<PreProcessTerm> pipeline);
  
  /** It would parse the input data and create the lexicon with frequency info. */
  public abstract Map<String, Integer> getTokensFreq();
  
  /** It would setup an index of the tokens from an external source. Note that Channel doesn't provide to create an index on its own.*/
	public abstract void setTokensIndex(Map<String, Integer> _tokens);


  public Map<Integer, Integer> getVector(String text) {
    text = this.tokeniser.parse(text);
    text = this.tokeniser.clean(text);
    String[] tokens = text.split("_");
    Map<Integer, Integer> inner = new HashMap<Integer, Integer>();
    for(String tok: tokens) {
      if(this.tokenIndex.containsKey(tok.trim())) {
        int tid = this.tokenIndex.get(tok);
        if(!inner.containsKey(tid))
          inner.put(tid, 1);
        else
          inner.put(tid, inner.get(tid)+1);
      }
    }
    return inner;
  }
  
  /** Parses the channel again and creates matrix based on the porivded index. The starting index is from 0.*/
	public abstract Map<Integer, Map<Integer, Integer>> getMatrix();
	
  public abstract Map<Integer, String> getDataIndex();
//	public String[] getRandomSentences(int n);
}
