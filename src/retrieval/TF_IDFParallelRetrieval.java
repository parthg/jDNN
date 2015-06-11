package retrieval;

import java.io.File;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.PrintWriter;

import java.util.List;
import java.lang.Exception;

import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import org.terrier.matching.ResultSet;
import es.upv.nlel.utils.Language;

import es.upv.nlel.utils.FileIO;
import es.upv.nlel.wrapper.TerrierWrapper;
import es.upv.nlel.corpus.FIRE;
import es.upv.nlel.corpus.Topic;
import es.upv.nlel.corpus.Topic.Tag;

import common.Dictionary;

import util.StringUtils;
import jair.GizaDict;

public class TF_IDFParallelRetrieval {
  TerrierWrapper terrier;
  String path_to_index="";
	TObjectIntHashMap<String> reverseMap = new TObjectIntHashMap<String>();	
	TIntObjectHashMap<String> map = new TIntObjectHashMap<String>();

  GizaDict gizaDict;
  Dictionary vocab;
  double[] idf;
  boolean partVocab = false;
 
  public String translateQuery(String q) throws IOException {
    List<String> terms = this.terrier.getTokens(q);
    String transQ = "";
    for(String t: terms) {
      if(this.gizaDict.contains(t.trim())) {
        if(this.partVocab && this.vocab.contains(t.trim()) && this.vocab.contains(this.gizaDict.get(t.trim()))) {
          int tid  = this.vocab.getId(t.trim());
          int tranTid = this.vocab.getId(this.gizaDict.get(t.trim()));
          if(this.idf[tid]>0.0 && this.idf[tranTid]>0.0)
            transQ += " " + this.gizaDict.get(t.trim());
        }
        else if(!this.partVocab)
          transQ += " " + this.gizaDict.get(t.trim());
      }

    }
    return transQ.trim();
  }

  public void loadVocab(String file) throws IOException {
    this.vocab = new Dictionary();
    this.vocab.load(file);
  }

  // Loads idf data calculated on the train parallel data
  public void loadIDF(String file) throws IOException {
    this.idf = new double[this.vocab.getSize()];
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
    String line = "";
    while((line = br.readLine())!=null) {
      String[] cols = line.split("\t");
      idf[Integer.parseInt(cols[0])] = Double.parseDouble(cols[1]);
    }
    br.close();
  }

  public static void main(String[] args) throws Exception {

    TF_IDFParallelRetrieval ret = new TF_IDFParallelRetrieval();
    String sentFile = "/home/parth/Dropbox/mt-autoencoder/sub-corpus/DNN-subparallel-hi-test-text.txt";

    /****************************************************/
//    String queryFile = "data/fire/joint/mt/output.0";
//    Language qLang = Language.HI;
  
    String queryFile = "/home/parth/Dropbox/mt-autoencoder/sub-corpus/DNN-subparallel-en-test-text.txt";
    Language qLang = Language.EN;

    ret.partVocab = true;
    ret.loadVocab("data/fire/joint/CL-LSI-dict.txt");
    ret.loadIDF("data/fire/joint/CL-LSI-idf.txt");
    /****************************************************/

    if(ret.partVocab)
      System.out.printf("Using only a part vocabulary of size %d.\n", ret.vocab.getSize());

    String corpus = "wmt-parallel-test";
    String lang = "hi";
    String dir = "output/temp-wmt-parallel-test/";
    boolean stopword_removal = true;
    boolean stem = false;
		String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		ret.terrier = new TerrierWrapper("/home/parth/workspace/terrier-3.5/");
		ret.path_to_index = "/home/parth/workspace/terrier-3.5/var/index/"+corpus+"/";

    int N = 1000;
    String matchModel = "TF_IDF";

    if(qLang == Language.EN) {
      ret.gizaDict = new GizaDict();
      ret.gizaDict.load("data/fire/joint/mt/lex.0.e2f", "data/fire/joint/mt/lex.0.f2e");
//      ret.gizaDict.load("data/fire/joint/mt/lex.0.e2f");
    }

    //sentences to files
//    ret.prepareFiles(sentFile, dir);
		
    // index if not present
		if(!new File(ret.path_to_index+"/"+lang+".docid.map").exists()) {
			if(!new File(ret.path_to_index).exists())
				new File(ret.path_to_index).mkdirs();
			ret.createIndex(ret.path_to_index, lang, "txt", dir, lang, stopword_removal, stem);
			ret.terrier.learnDocId(path_to_terrier +"etc/collection.spec");
		}
		
		// Setting up the Pipeline
    ret.terrier.setLanguage(lang);
		if(stopword_removal)
			ret.terrier.setStopwordRemoval(lang);
		if(stem)
			ret.terrier.setStemmer(lang);
	
    // load the index  
		ret.terrier.setIndex(ret.path_to_index, lang);
		ret.loadEnDocs(lang, corpus);
		ret.terrier.loadIndex(ret.path_to_index, lang, lang);
		
		int dim = ret.terrier.getDimension();
    System.out.printf("Total terms = %d\n", dim);

    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(queryFile), "UTF-8"));
    String line = "";
    double mrr = 0.0;
    int count =1;
    while((line = br.readLine())!=null) {
      String q = "";
      if(qLang == Language.EN) {
        q = line.trim();
        q = ret.translateQuery(q);
      }
      else if(qLang == Language.HI)
        q = ret.terrier.pipelineText(line);
      if(q.length()>1) {
        ResultSet rs = ret.terrier.getResultSet(q, matchModel, true, 0);
        int[] docid = rs.getDocids();
        double[] scores = rs.getScores();
        int pos = 1;
        for(int i=0; i<docid.length; i++) {
          int id = Integer.parseInt(StringUtils.removeExt(StringUtils.fileName(ret.map.get(docid[i]))));
          if(count == id) {
            mrr+=1.0/pos;
            break;
          }
          else pos++;
        }
      }
      count++;
    }
    System.out.printf("MRR = %.6f\n", mrr/count);
    br.close();
  }
  
	
  public void createIndex(String path_to_index, String prefix, String ext, String path_to_data,
			String lang, boolean stopword_removal, boolean stem) throws IOException {
		this.terrier.setIndex(path_to_index, prefix);
		this.terrier.prepareIndex(path_to_data, ext, lang, stopword_removal, stem);
	}
	
  public void loadEnDocs(String lang, String corpus) throws IOException {
		this.map = this.terrier.learnDocName(this.path_to_index+lang+".docid.map");
		for(int i: this.map.keys()) {
			this.reverseMap.put(this.map.get(i), i);
		}
	}

  public void prepareFiles(String file, String outDir) throws IOException {
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
    String line = "";
    int count = 1;
    while((line = br.readLine())!=null) {
      FileIO.stringToFile(new File(outDir+count+".txt"), line.trim(), false);
      count++;
    }
    br.close();
  }
}
