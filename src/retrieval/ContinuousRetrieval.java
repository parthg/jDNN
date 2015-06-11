package retrieval;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

import common.Sentence;
import common.Corpus;
import common.Dictionary;
import common.RankList;
import common.Qrel;
import models.Model;
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

public class ContinuousRetrieval {
  Qrel qrel;
  Model qModel, dModel;
  Language qLang, dLang;
  List<Topic> topics;
  DMatrix qVocab, dVocab;
  Corpus corpus;
  DMatrix index;
  DMatrix topicsMat;
  DMatrix sim;
  Map<String, Double> idf;
  
  // load Qrel
  public void loadQrel(String file) throws IOException {
    this.qrel = new Qrel(file);
  }

  // load dict
  public Dictionary loadDict(String file) throws IOException {
    Dictionary dict = new Dictionary();
    dict.load(file);
    return dict;
  }
  
  public void loadIdf(String file) throws IOException {
    this.idf = new HashMap<String, Double>();
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
    String line = "";
    while((line = br.readLine())!=null) {
      String[] cols = line.split("\t");
      String term = cols[1].trim();
      Double val = Double.parseDouble(cols[2].trim());
      this.idf.put(term, val);
    }
    br.close();
  }

  public void updateDictIdf() {
    for(int i=0; i<this.qModel.dict().getSize(); i++) {
      String term = this.qModel.dict().getTerm(i);
      if(this.idf.containsKey(term))
        this.qModel.dict().setIdf(term, this.idf.get(term));
    }

    System.out.printf("IDF of term %s = %.6f\n", this.qModel.dict().getTerm(550), this.qModel.dict().getIdf(550));

    for(int i=0; i<this.dModel.dict().getSize(); i++) {
      String term = this.dModel.dict().getTerm(i);
      if(this.idf.containsKey(term))
        this.dModel.dict().setIdf(term, this.idf.get(term));
    }
    System.out.printf("IDF of term %s = %.6f\n", this.dModel.dict().getTerm(550), this.dModel.dict().getIdf(550));
  }
  // load both models
  public void loadModel(String qModelFile, Dictionary qDict, String dModelFile, Dictionary dDict) throws IOException {
    this.qModel = new AddModel();
    this.qModel.load(qModelFile, qDict);

    this.dModel = new AddModel();
    this.dModel.load(dModelFile, dDict);
  }

  // project vocab
  public void projectVocabulary() {
/*    this.qVocab = this.qModel.projectVocabulary(5000);
    this.dVocab = this.dModel.projectVocabulary(5000);*/
    
    this.qVocab = this.qModel.projectVocabularyWithIdf(5000);
    this.dVocab = this.dModel.projectVocabularyWithIdf(5000);
  }
/*  // load model
  public void loadModel(String file, Dictionary dict) throws IOException {
    this.model = new AddModel();
    this.model.load(file, dict);
  }

  // project vocab
  public void projectVocabulary() {
    this.vocab = this.model.projectVocabulary(5000);
  }*/

  // load collection
  public void loadCollection(String file) throws IOException {
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, this.dLang, path_to_terrier, pipeline);
		this.corpus = new Corpus();
    this.corpus.load(file, true, ch, this.dModel.dict(), false);
  }

  // prepare index
  public void prepareIndex() {
    this.index = DMath.createMatrix(this.corpus.getSize(), this.dModel.outSize());
    for(int i=0; i<this.corpus.getSize(); i++) {
      Sentence s = this.corpus.get(i);
//      System.out.println(s.toString());
      DMatrix row = DMath.createZerosMatrix(1, this.dModel.outSize());
      row.muli(0.0);
      for(int j=0; j<s.getSize(); j++) {
//        System.out.println("Term = " + s.get(j));
//        this.vocab.getRow(s.get(j)).print();
        // TODO: This step might be huge overhead by using GPU
        row.addi(this.dVocab.getRow(s.get(j)));
      }
      this.index.fillRow(i, row);
    }
  }

  // load queries
  public void loadQueries(String file) throws IOException {
    this.topics = FIRE.parseTopicFile(file);
    this.topicsMat = DMath.createMatrix(this.topics.size(), this.qModel.outSize());
    
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, this.qLang, path_to_terrier, pipeline);
    
    int i=0;
    for(Topic t: this.topics) {
      DMatrix row = DMath.createZerosMatrix(1, this.qModel.outSize());
      row.muli(0.0);
      String text = t.get(Tag.TITLE);
      text = ch.getTokeniser().parse(text);
      text = ch.getTokeniser().clean(text);
      String[] tokens = text.split("_");
      for(String tok: tokens) {
        if(this.qModel.dict().contains(tok)) {
          // TODO: Might not be so efficient with GPU
          row.addi(this.qVocab.getRow(this.qModel.dict().getId(tok)));
        }
      }
      this.topicsMat.fillRow(i, row);
      i++;
    }
  }
  
  // retrieve
  public void similarity() {
    DMatrix topicsNorm = this.topicsMat.vectorNorm();
    this.topicsMat.getRow(0).print("Topic1 Raw");
    topicsNorm.getRow(0).print("Topic1");
    this.index.getRow(0).print("Doc1 Raw");
    DMatrix indexNorm = this.index.vectorNorm();
    indexNorm.getRow(0).print("Doc1");
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
    ContinuousRetrieval ret = new ContinuousRetrieval();
//    String modelPrefix = "tanh-hi-dict-400-b-100-h-128-new";

    ret.qLang = Language.EN;
    ret.dLang = Language.HI;
    
    
//    Dictionary dict = ret.loadDict("obj/"+modelPrefix+"/dict.txt");
    Dictionary qDict = ret.loadDict("data/fire/en/dict-100.txt");
    Dictionary dDict = ret.loadDict("data/fire/hi/dict-400.txt");
/*    Dictionary qDict = ret.loadDict("data/fire/en/dict-parallel.txt");
    Dictionary dDict = ret.loadDict("data/fire/hi/dict-titles-full.txt");*/
    System.out.printf("Dictionary Loaded.\n");

    int iter = 78;

    String runPrefix = "cl-b-100-h-128-T-2011-12-10k";

//    String qModelFile = "obj/tanh-cl-w-0.1-40k-b-100-h-128/model_iter81.txt"; // for full vocab Q
    String qModelFile = "obj/tanh-cl-w-0.1-b-100-h-128/model_iter78.txt"; // Best so far for 10k Q
//    String dModelFile = "obj/tanh-b-100-h-128/model_iter14.txt";

//    String dModelFile = "obj/tanh-b-100-h-128/model_iter14.txt"; // for full vocab D
//    String qModelFile = "obj/tanh-hi-dict-400-b-100-h-128-new/model_iter20.txt";
    String dModelFile = "obj/tanh-hi-dict-400-b-100-h-128-new/model_iter20.txt"; // Best so far for 10k D

//    ret.loadModel("obj/"+modelPrefix+"/model_iter"+iter+".txt", dict);
    ret.loadModel(qModelFile, qDict, dModelFile, dDict);
    System.out.printf("Model Loaded.\n");

    ret.loadIdf("data/fire/joint/joint-idf.txt");
    ret.updateDictIdf();
    
    ret.projectVocabulary();
    System.out.printf("Vocabulary Projected.\n");

/*    DMatrix term1 = DMath.createMatrix(1, dict.getSize());
    term1.put(2759, 1.0);


    DMatrix term2 = DMath.createMatrix(1, dict.getSize());
    term2.put(2963, 1.0);

    ret.model.fProp(term1).print("Ideal 2759");
    ret.model.fProp(term2).print("Ideal 2963");

    ret.vocab.getRow(2759).print("with projected vocab: 2957");
    ret.vocab.getRow(2963).print("with projected vocab: 2963");*/
    ret.loadCollection("etc/data/fire/hi/title-only-with-did.txt");
//    ret.loadCollection("sample/title-only-sample.txt");
    System.out.printf("Collection Loaded.\n");

    ret.prepareIndex();
    System.out.printf("Index Prepared.\n");
//    ret.index.print();

//    ret.loadQueries("/home/parth/workspace/data/fire/topics/en.topics.126-175.2011.txt");
    ret.loadQueries("/home/parth/workspace/data/fire/topics/en.topics.2011-12.txt");
//    ret.loadQueries("/home/parth/workspace/data/fire/topics/en.topics.176-225.2012.txt");
    System.out.printf("Topics Loaded.\n");
//    ret.topicsMat.print();

    ret.similarity();
    System.out.printf("Similarity Estimated.\n");

    ret.printRankList("output/rl-cl-dnn-title-idf"+"-"+runPrefix+".txt",1000);
//    ret.sim.print();

  }
}
