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
import models.Model;
import models.S2Net;
import models.Autoencoder;

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
  Model nnModel;
  List<Topic> topics;
  DMatrix vocab;
  Corpus corpus;
  DMatrix index;
  DMatrix topicsMat;
  DMatrix sim;
  double[] idf;
  String modelType;  
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

  public void loadAE(String file) throws IOException {
    this.nnModel = new Autoencoder();
    this.nnModel.load(file, this.dict);
  }
  public void loadS2Net(String file) throws IOException {
    this.nnModel = new S2Net();
    this.nnModel.load(file, this.dict);
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
    int modelInDim =0, modelOutDim = 0;
    if(this.modelType.equals("S2Net") || this.modelType.equals("AE")) {
      this.index = DMath.createMatrix(this.corpus.getSize(), this.nnModel.outSize());
      modelInDim = this.nnModel.inSize();
      modelOutDim = this.nnModel.outSize();
    }
    else {
      this.index = DMath.createMatrix(this.corpus.getSize(), this.model.outSize());
      modelInDim = this.model.inSize();
      modelOutDim = this.model.outSize();
    }
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
      System.out.printf("Model dimension = %d x %d and Batch dimension %d x %d\n", modelInDim, modelOutDim, batch.rows(), batch.columns());
//      DMatrix proj = this.model.project(batch);
      DMatrix proj = null;
      if(this.modelType.equals("S2Net") || this.modelType.equals("AE"))
        proj = this.nnModel.fProp(batch);
      else
        proj = this.model.project(batch);
      this.index.fillMatrix(i, proj);
      i+=batchSize;
      
    }
  }

  // load queries
  public void loadQueries(String file) throws IOException {
    this.topics = FIRE.parseTopicFile(file);
    int modelOutDim=0, modelInDim=0;
    if(this.modelType.equals("S2Net") || this.modelType.equals("AE")) {
      this.topicsMat = DMath.createMatrix(this.topics.size(), this.nnModel.outSize());
      modelInDim = this.nnModel.inSize();
      modelOutDim = this.nnModel.outSize();
    }
    else {
      this.topicsMat = DMath.createMatrix(this.topics.size(), this.model.outSize());
      modelInDim = this.model.inSize();
      modelOutDim = this.model.outSize();
    }

    DMatrix topicRep = DMath.createMatrix(this.topics.size(), this.dict.getSize());
    
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, Language.EN, path_to_terrier, pipeline);
    
    int i=0;
    for(Topic t: this.topics) {
      DMatrix row = DMath.createMatrix(1, modelOutDim);
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
//    this.topicsMat = this.model.project(topicRep);
    if(this.modelType.equals("S2Net") || this.modelType.equals("AE"))
      this.topicsMat = this.nnModel.fProp(topicRep);
    else
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

    ret.modelType = "CL-LSI";
    String modelPrefix = "CL-LSI-T-2011-12";
    String dictFile = "data/fire/joint/CL-LSI-dict.txt";
    if(ret.modelType.equals("AE"))
      dictFile = "data/fire/joint/AE-dict.txt";
    ret.loadDict(dictFile);
    if(ret.modelType.equals("AE"))
      ret.loadIDF("data/fire/joint/AE-idf.txt");
    else
      ret.loadIDF("data/fire/joint/CL-LSI-idf.txt");
//    Dictionary dict = ret.loadDict("data/fire/hi/dict-400.txt");
    System.out.printf("Dictionary Loaded.\n");

    String modelFile = "data/fire/joint/ProjMat-CL-LSI.mat";
//    String modelFile = "data/fire/joint/ProjMat-OPCA-0.1.mat";
//    String modelFile = "obj/s2net-cl-h-128-new/model_iter8.txt";
//    String modelFile = "scripts/matlab/try"; // THIS IS AE
    if(ret.modelType.equals("CL-LSI") || ret.modelType.equals("OPCA")) {
      ret.loadModel(new File(modelFile));
    }
    else if(ret.modelType.equals("S2Net")) {
      ret.loadS2Net(modelFile);
    }
    else if(ret.modelType.equals("AE")) {
      ret.loadAE(modelFile);
    }
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

//    ret.loadQueries("/home/parth/workspace/data/fire/topics/en.topics.126-175.2011.txt");
    ret.loadQueries("/home/parth/workspace/data/fire/topics/en.topics.2011-12.txt");
    System.out.printf("Topics Loaded.\n");
//    ret.topicsMat.print();

    ret.similarity();
    System.out.printf("Similarity Estimated.\n");

    ret.printRankList("output/rl-cl-"+modelPrefix+".txt",1000);
//    ret.sim.print();

  }
}
