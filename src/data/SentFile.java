package data;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import es.upv.nlel.corpus.NEWSDocType;


public class SentFile extends Channel {
  public static int MIN_WORD_LENGTH = 3;
  String dataPath;
  public SentFile(String path_to_data) {
    this.dataPath = path_to_data;
  }
  public void setParser(NEWSDocType type) {
  	// note required for this.
  }

  public Map<String, Integer> getTokensFreq() {
    this.tokenFreq = new HashMap<String, Integer>();
    BufferedReader br = null;
		try {
			br = new BufferedReader(new InputStreamReader(new FileInputStream(this.dataPath), "UTF-8"));
			String line = "";
			while((line = br.readLine())!=null) {
	     	String text = line;
	     	text = this.tokeniser.parse(text);
				text = this.tokeniser.clean(text);
				String[] tokens = text.split("_");
				for(String tok: tokens) {
					if(tok.trim().equals("N") || tok.trim().length()>MIN_WORD_LENGTH) {
						if(!this.tokenFreq.containsKey(tok.trim()))
							this.tokenFreq.put(tok.trim(), 1);
						else
							this.tokenFreq.put(tok.trim(), this.tokenFreq.get(tok.trim())+1);
					}
				}
	    }
			br.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
    return this.tokenFreq;
  }

  public Map<Integer, Map<Integer, Integer>> getMatrix() {
  	Map<Integer, Map<Integer, Integer>> matrix = new HashMap<Integer, Map<Integer, Integer>>();
  	this.docIndex = new HashMap<Integer, String>();
  	 BufferedReader br = null;
 		try {
 			br = new BufferedReader(new InputStreamReader(new FileInputStream(this.dataPath), "UTF-8"));
 			String line = "";
 			int docid = 0;
 			while((line = br.readLine())!=null) {
 				String text = line;

        Map<Integer, Integer> inner = this.getVector(text);
 				if(inner.size()>=2) {
 					matrix.put(docid, inner);
 					this.docIndex.put(docid, line.trim());
 					docid++;
 				}
 			}
 			br.close();
    }
 		catch(Exception e) {
 			e.printStackTrace();
 		}
  	return matrix;
  }
}
