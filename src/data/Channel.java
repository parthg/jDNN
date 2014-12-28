package data;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.IOException;

import es.upv.nlel.corpus.NEWSDocType;
import es.upv.nlel.utils.Language;

/** Channel Abstract class is responsible for input channel of data. Input channel can be a directory for which all the files with a particular extension or a per-line sentence file.
 * To use it, fist make an instance of it with the implementing class (DocCollection or SentFile), setup and setParser.
 *
 * @author Parth Gupta
 */
public abstract class Channel {
  Map<String, Integer> tokenFreq;
	Map<String, Integer> tokenIndex;
	Map<Integer, String> docIndex;
	Tokeniser tokeniser;
	Language lang;
	public abstract void setParser(NEWSDocType type);
	
	public void setup(TokenType tokenType, Language _lang, String path_to_terrier, List<PreProcessTerm> pipeline) {
  	this.lang = _lang;
  	this.tokeniser = TokeniserFactory.getTokeniser(tokenType);
  	this.tokeniser.setup(_lang, path_to_terrier, pipeline);
  }

  public Tokeniser getTokeniser() {
    return this.tokeniser;
  }
  
  /** It would parse the input data and create the lexicon with frequency info. */
  public abstract Map<String, Integer> getTokensFreq();
  
  /** It would setup an index of the tokens from an external source. Note that Channel doesn't provide to create an index on its own.*/
  public void setTokensIndex(Map<String, Integer> _tokens) {
  	this.tokenIndex = _tokens;
  }

  public void loadTokenIndex(String indexFile) throws IOException{
    this.tokenIndex = new HashMap<String, Integer>();
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(indexFile),"UTF-8"));
    String line = "";
    while((line = br.readLine())!=null) {
      String[] cols = line.split("\t");
      this.tokenIndex.put(cols[0].trim(), Integer.parseInt(cols[1].trim()));
    }
    br.close();
  }

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
	
  public Map<Integer, String> getDataIndex() {
    return this.docIndex;
  }
}
