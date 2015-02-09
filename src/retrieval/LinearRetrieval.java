package retrieval;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.FileNotFoundException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.File;

import common.Sentence;
import common.Corpus;
import common.Dictionary;
import common.RankList;
import common.Qrel;
import models.LinearModel;
import models.AddModel;

import math.DMath;
import math.DMatrix;

import util.StringUtils;

import data.Channel;
import data.SentFile;
import data.TokenType;
import data.PreProcessTerm;

import es.upv.nlel.corpus.FIRE;
import es.upv.nlel.corpus.Topic;
import es.upv.nlel.corpus.Topic.Tag;
import es.upv.nlel.utils.Language;

public class LinearRetrieval {
  Qrel qrel;
  Dictionary dict;
  LinearModel model;
  List<Topic> topics;
  DMatrix vocab;
  Corpus corpus;
  DMatrix index;
  DMatrix topicsMat;
  DMatrix sim;
  double[] idf;
  
  // load Qrel
  public void loadQrel(String file) throws IOException {
    this.qrel = new Qrel(file);
  }

  // load dict
  public void loadDict(String file) throws IOException {
    this.dict = new Dictionary();
    this.dict.load(file);
  }
  
  // load model
  public void loadModel(File file) throws IOException {
    this.model = new LinearModel(this.dict.getSize(), 128);
    this.model.load(file);
  }

/*  // project vocab
  public void projectVocabulary() {
    this.vocab = this.model.projectVocabulary(5000);
  }*/

  public void loadIDF(String file) throws IOException {
    this.idf = new double[this.dict.getSize()];
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
    String line = "";
    while((line = br.readLine())!=null) {
      String[] cols = line.split("\t");
      idf[Integer.parseInt(cols[0])] = Double.parseDouble(cols[1]);
    }
    br.close();
  }

  // load collection
  public void loadCollection(String file) throws IOException {
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
		this.corpus = new Corpus();
    this.corpus.load(file, true, ch, this.dict, false);
  }

  // prepare index
  public void prepareIndex() {
    this.index = DMath.createMatrix(this.corpus.getSize(), this.model.outSize());
    int batchSize = 10000;
    for(int i=0; i<this.corpus.getSize(); ) {
      if(i+batchSize > this.corpus.getSize()) {
        batchSize = this.corpus.getSize() - i;
      }
      DMatrix batch = DMath.createMatrix(batchSize, this.dict.getSize());
      for(int j=0; j<batchSize; j++) {
        Sentence s = this.corpus.get(i+j);
        Map<Integer, Integer> docTf = new HashMap<Integer, Integer>();
        for(int k=0; k<s.getSize(); k++) {
          if(docTf.containsKey(s.get(k)))
            docTf.put(s.get(k), docTf.get(s.get(k))+1);
          else
            docTf.put(s.get(k), 1);
        }
        for(int k : docTf.keySet()) {
          double v = (Math.log(1.0+(double)docTf.get(k))/Math.log(2.0))*idf[k];
          batch.put(j, k, v);
        }
      }
      DMatrix proj = this.model.project(batch);
      this.index.fillMatrix(i, proj);
      i+=batchSize;
      
    }
  }

  // load queries
  public void loadQueries(String file) throws IOException {
    this.topics = FIRE.parseTopicFile(file);
    this.topicsMat = DMath.createMatrix(this.topics.size(), this.model.outSize());

    DMatrix topicRep = DMath.createMatrix(this.topics.size(), this.dict.getSize());
    
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
    
    int i=0;
    for(Topic t: this.topics) {
      DMatrix row = DMath.createMatrix(1, this.model.outSize());
      String text = t.get(Tag.TITLE);
      text = ch.getTokeniser().parse(text);
      text = ch.getTokeniser().clean(text);
      String[] tokens = text.split("_");
      Map<Integer, Integer> docTf = new HashMap<Integer, Integer>();
      for(String tok: tokens) {
        if(this.dict.contains(tok)) {
          if(docTf.containsKey(this.dict.getId(tok)))
            docTf.put(this.dict.getId(tok), docTf.get(this.dict.getId(tok))+1);
          else
            docTf.put(this.dict.getId(tok), 1);
        }
      }
      for(int k: docTf.keySet()) {
        double v = (Math.log(1.0+(double)docTf.get(k))/Math.log(2.0))*idf[k];
        topicRep.put(i, k, v);
      }
      i++;
    }
    this.topicsMat = this.model.project(topicRep);
  }
  
  // retrieve
  public void similarity() {
    DMatrix topicsNorm = this.topicsMat.vectorNorm();
    DMatrix indexNorm = this.index.vectorNorm();
    this.sim = topicsNorm.mmul(false, true, indexNorm);
  }

  public void printRankList(String outFile, int N) throws FileNotFoundException{
    PrintWriter p = new PrintWriter(outFile);
    for(int i=0; i<this.sim.rows(); i++) {
      int[] rl = RankList.rankList(this.sim.getRow(i).data(), N);
      int qid = this.topics.get(i).getID();
      for(int j=0; j<rl.length; j++) {
        p.println(qid+" Q0 "+this.corpus.get(rl[j]).label()+" "+(j+1)+" "+ this.sim.get(i, rl[j]));
      }
    }
    p.close();
    // get the ranklist of the docid sent and then print it in the TREC format taking help from Corpus
  }

  // evaluate
  //
  public static void main(String[] args) throws IOException {
    LinearRetrieval ret = new LinearRetrieval();
/*    String qrelFile = "sample/sample-qrel.txt";
    ret.loadQrel(qrelFile);
    System.out.printf("Qrel Loaded.\n");*/

    String modelPrefix = "tanh-hi-dict-70k-b-150-h-128";
    ret.loadDict("obj/"+modelPrefix+"/dict.txt");
//    Dictionary dict = ret.loadDict("data/fire/hi/dict-400.txt");
    System.out.printf("Dictionary Loaded.\n");

    int iter = 8;

    ret.loadModel(new File("obj/"+modelPrefix+"/model_iter"+iter+".txt"));
    System.out.printf("Model Loaded.\n");

/*    ret.projectVocabulary();
    System.out.printf("Vocabulary Projected.\n");*/

/*    DMatrix term1 = DMath.createMatrix(1, dict.getSize());
    term1.put(2759, 1.0);


    DMatrix term2 = DMath.createMatrix(1, dict.getSize());
    term2.put(2963, 1.0);

    ret.model.fProp(term1).print("Ideal 2759");
    ret.model.fProp(term2).print("Ideal 2963");

    ret.vocab.getRow(2759).print("with projected vocab: 2957");
    ret.vocab.getRow(2963).print("with projected vocab: 2963");*/
    ret.loadCollection("etc/data/fire/hi/title-only-with-did.txt");
    System.out.printf("Collection Loaded.\n");

    ret.prepareIndex();
    System.out.printf("Index Prepared.\n");
//    ret.index.print();

    ret.loadQueries("/home/parth/workspace/data/fire/topics/hi.topics.126-175.2011.txt");
    System.out.printf("Topics Loaded.\n");
//    ret.topicsMat.print();

    ret.similarity();
    System.out.printf("Similarity Estimated.\n");

    ret.printRankList("output/rl-"+modelPrefix+"-iter-"+iter+".txt",1000);
//    ret.sim.print();

  }
}
