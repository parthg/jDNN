package data;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import es.upv.nlel.corpus.NEWSDocType;
import es.upv.nlel.parser.FIREParserFactory;
import es.upv.nlel.parser.FIREParserInterface;
import es.upv.nlel.utils.FileIO;
import es.upv.nlel.utils.Language;
import es.upv.prhlt.sentence.Splitter;
import es.upv.prhlt.sentence.SplitterFactory;
import random.RandomUtils;

public class RandSentences implements Channel {
  Map<String, Integer> tokenFreq;
  Map<String, Integer> tokenIndex;
  Map<Integer, String> docIndex;
  String dataPath;
  Tokeniser tokeniser;
  FIREParserInterface parser;
  String ext;
  Language lang;
  int sampleSize=-1;
  List<String> sentences;
  List<String> randSent;
  public RandSentences(String path_to_data, String _ext, int _sampleSize) {
    this.dataPath = path_to_data;
    this.ext = _ext;
    this.sampleSize = _sampleSize;
  }
  public void setParser(NEWSDocType type) {
  	this.parser = FIREParserFactory.getParser(type);
  }
  public void setup(TokenType tokenType, Language _lang, String path_to_terrier, List<PreProcessTerm> pipeline) {
  	this.lang = _lang;
  	this.tokeniser = TokeniserFactory.getTokeniser(tokenType);
  	this.tokeniser.setup(_lang, path_to_terrier, pipeline);
  }
  /** In this case the docindex is actually the sentence itself mapped to the id.
   * 
   * @return
   */
  public Map<Integer, String> getDataIndex() {
    return this.docIndex;
  }
  public Map<String, Integer> getTokensFreq() {
    this.tokenFreq = new HashMap<String, Integer>();
    this.sentences = new ArrayList<String>();
  	this.randSent = new ArrayList<String>();
  	if(this.sampleSize == -1) {
  		System.out.println("The Sample size is not specified.. hence choosine 100k..");
  		this.sampleSize = 100000;
  	}
  	
  	try {
  		Splitter splitter = SplitterFactory.getSplitter(this.lang);
  		List<String> files = FileIO.getFilesRecursively(new File(this.dataPath), ".txt");
      for(String f: files) {
      	String text = this.parser.parse(f).getContent();
      	String[] sents = splitter.getSentences(text);
      	sentences.addAll(Arrays.asList(sents));
      }
//      if(this.sampleSize==0)
//        return sentences.toArray(new String[sentences.size()]);
      int ceil = sentences.size();
      System.out.printf("Total Sentences = %d and now selecting %d samples from it", ceil, this.sampleSize);
      RandomUtils.seed(1234L);
      int[] randIndex = RandomUtils.getRandomSamples(ceil, this.sampleSize);
      for(int i=0; i<randIndex.length; i++)
        randSent.add(sentences.get(randIndex[i]));
      for(String text: randSent) {
  			text = this.tokeniser.parse(text);
  			text = this.tokeniser.clean(text);
  			String[] tokens = text.split("_");
  			for(String tok: tokens) {
  				if(tok.trim().equals("N") || tok.trim().length()>2) {
  					if(!this.tokenFreq.containsKey(tok.trim()))
  						this.tokenFreq.put(tok.trim(), 1);
  					else
  						this.tokenFreq.put(tok.trim(), this.tokenFreq.get(tok.trim())+1);
  				}
  			}
      }
    }
  	catch(Exception e) {
  		e.printStackTrace();
  	}
    return this.tokenFreq;
  }
  public void setTokensIndex(Map<String, Integer> _tokens) {
  	this.tokenIndex = _tokens;
  }
  public Map<Integer, Map<Integer, Integer>> getMatrix() {
    try {
      if(this.tokenIndex.equals(null))
        System.out.println("Please set an index first..");
    }
    catch (NullPointerException e) {
      System.out.println("Please set an index first..");
      e.printStackTrace();
    }
  	Map<Integer, Map<Integer, Integer>> matrix = new HashMap<Integer, Map<Integer, Integer>>();
    this.docIndex = new HashMap<Integer, String>();
  	int docid = 0;
  	for(String sent: randSent) {
      sent = sent.trim();
			String text = this.tokeniser.parse(sent);
			text = this.tokeniser.clean(text);
			String[] tokens = text.split("_");
			Map<Integer, Integer> inner = new HashMap<Integer, Integer>();
			for(String tok: tokens) {
				if(tokenIndex.containsKey(tok.trim())) {
					int tid = this.tokenIndex.get(tok);
					if(!inner.containsKey(tid))
						inner.put(tid, 1);
					else
						inner.put(tid, inner.get(tid)+1);
				}
			}
			if(inner.size()>=2) {
				matrix.put(docid, inner);
				this.docIndex.put(docid, sent.trim());
				docid++;
			}
    }
  	return matrix;
  }
}
