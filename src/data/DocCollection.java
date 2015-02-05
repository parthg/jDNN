package data;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

import es.upv.nlel.corpus.NEWSDocType;
import es.upv.nlel.parser.FIREParserFactory;
import es.upv.nlel.parser.FIREParserInterface;
import es.upv.nlel.utils.FileIO;

public class DocCollection extends Channel {
  String dataPath;
  FIREParserInterface parser;
  String ext;
  List<String> titles;
  public DocCollection(String path_to_data, String _ext) {
    this.dataPath = path_to_data;
    this.ext = _ext;
  }
  public void setParser(NEWSDocType type) {
  	this.parser = FIREParserFactory.getParser(type);
  }
  public Map<String, Integer> getTokensFreq() {
    this.titles = new ArrayList<String>();
    this.tokenFreq = new HashMap<String, Integer>();
    List<String> files = FileIO.getFilesRecursively(new File(this.dataPath), ext);
    for(String f: files) {
    	String text = "";
    
//			if(f.toLowerCase().contains("navbharattimes"))
			text = this.parser.parse(f).getTitle();
      this.titles.add(text);

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
    return this.tokenFreq;
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
  	List<String> files = FileIO.getFilesRecursively(new File(this.dataPath), ".txt");
  	int docid = 0;
    System.out.printf("\t\tThere are total %d files.\n",files.size());
    for(String f: files) {
      if(docid%20000==0)
        System.out.printf("Processed files: %d\n",docid);
			String title = this.parser.parse(f).getTitle().trim();
			
      Map<Integer, Integer> inner = this.getVector(title);
			if(inner.size()>=2) {
				matrix.put(docid, inner);
				this.docIndex.put(docid, new File(f).getName()+"__"+title);
				docid++;
			}
    }
  	return matrix;
  }

  public List<String> titles() {
    return this.titles;
  }
}
