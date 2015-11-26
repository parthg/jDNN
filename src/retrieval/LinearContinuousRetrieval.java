package retrieval;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.FileNotFoundException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;

import common.Sentence;
import common.Corpus;
import common.Dictionary;
import common.RankList;
import common.Qrel;
import models.LinearModel;
import models.AddModel;
import models.Model;

import math.DMath;
import math.DMatrix;

import util.StringUtils;

import data.Channel;
import data.SentFile;
import data.TokenType;
import data.PreProcessTerm;

import es.upv.nlel.parser.FIREDoc;
import es.upv.prhlt.sentence.Splitter;
import es.upv.prhlt.sentence.SplitterFactory;
import es.upv.nlel.corpus.FIRE;
import es.upv.nlel.corpus.Topic;
import es.upv.nlel.corpus.Topic.Tag;
import es.upv.nlel.utils.Language;
import es.upv.nlel.utils.FileIO;
import es.upv.nlel.parser.FIREParserInterface;
import es.upv.nlel.parser.FIREParserFactory;
import es.upv.nlel.corpus.NEWSDocType;


public class LinearContinuousRetrieval {
  NEWSDocType docType;
  FIREParserInterface parser;
  Qrel qrel;
  LinearModel model;
  Dictionary dict;
  List<Topic> topics;
  DMatrix vocab;
  Corpus corpus;
  DMatrix index;
  DMatrix topicsMat;
  DMatrix sim;
  Language queryLang, docLang;
  double[] idf;
  String[] ids;
  boolean useTitle = true;
  boolean useContent = false;
  boolean useBoWDoc = false;


  String granularity = "sent"; // "doc"
  
  // load Qrel
  public void loadQrel(String file) throws IOException {
    this.qrel = new Qrel(file);
  }

  // load dict
  public void loadDict(String file) throws IOException {
    this.dict = new Dictionary();
    this.dict.load(file);
  }
  
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

/*  // load model
  public void loadModel(String file, Dictionary dict) throws IOException {
    this.model = new AddModel();
    this.model.load(file, dict);
  }*/

  public void loadModel(File file) throws IOException {
    this.model = new LinearModel(this.dict.getSize(), 128);
    this.model.load(file);
  }


/*  // project vocab
  public void projectVocabulary() {
    this.vocab = this.model.projectVocabulary(10000);
  }*/

  // load collection with sentence file, each document is a line in the file
  public void loadCollection(File file) throws IOException {
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile(file.getAbsolutePath());
		ch.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
		this.corpus = new Corpus();
    this.corpus.load(file.getAbsolutePath(), true, ch, this.dict, false);
  }

/*  public void loadCollection(String dir) throws IOException {
    this.batches = new ArrayList<DoubleMatrix>();
    this.sentCount = new ArrayList<List<Integer>>();
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);

    Splitter splitter = SplitterFactory.getSplitter(this.lang);
    List<String> files = FileIO.getFilesRecursively(new File(dir), ".txt");

    DMatrix batch = DMath.createMatrix(this.batchSize, this.dict.getSize());
    List<String> ids = new ArrayList<String>();
    for(int f=0; f< files.size(); f++) {
      FIREDoc doc = his.parser.parse(files.get(f));
      String text = doc.getTitle();
      //TODO: Proper sentence split delimeter
      text = text + " । " + doc.getContent();
      text = this.tokeniser.parse(text);
      text = this.tokeniser.clean(text);
      String[] tokens = text.split("_");
      DMatrix dVec = DMath.createZerosMatrix(1, this.dict.getSize());
      ids.add(f.getName());
      for(String tok: tokens) {
        if(this.dict.contains(tok)) {
          dVec.put(this.dict.getId(tok), dVec.get(this.dict.getId(tok))+1.0);
        }
      }
      if(f%this.batchSize == 0 && f>0) {
        DMatrix h = this.projectData(batch);
        this.batches.add(h);
        this.docids.add(ids);
        batch = DMath.createMatrix(this.batchSize, this.dict.getSize());
        batch.fillRow(f%this.batchSize, dVec);
        ids = new ArrayList<String>();
      }
      else if(f == files.size()-1) {
        DMatrix h = this.projectData(batch);
        this.batches.add(h);
        this.docids.add(ids);
      } 
      else {
        batch.fillRow((f%this.batchSize), dVec);
      }
    }
    System.out.printf("Finally loaded %d files\n", files.size());
  }*/

  public DMatrix getRepresentation(Channel ch, String text) {
    text = ch.getTokeniser().parse(text);
    text = ch.getTokeniser().clean(text);
    String[] tokens = text.split("_");
    DMatrix dVec = DMath.createZerosMatrix(1, this.dict.getSize());
    boolean use = false;
      
    
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
      use = true;
      double v = (Math.log(1.0+(double)docTf.get(k))/Math.log(2.0))*this.idf[k];
      dVec.put(k, v);
    }
      
/*    for(String tok: tokens) {
      if(this.dict.contains(tok)) {
        use = true;
        dVec.put(this.dict.getId(tok), dVec.get(this.dict.getId(tok))+1.0);
      }
    }*/
    if(use)
      return dVec;
    else
      return null;
  }

  public void calculateSimilarity(String dir) throws IOException {
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.remove(PreProcessTerm.SW_REMOVAL);
		pipeline.remove(PreProcessTerm.STEM);

		Channel ch = new SentFile("");
		ch.setup(TokenType.WORD, this.docLang, path_to_terrier, pipeline);

    this.parser = FIREParserFactory.getParser(this.docType);
    
    Splitter splitter = SplitterFactory.getSplitter(this.docLang);
    List<String> files = FileIO.getFilesRecursively(new File(dir), ".txt");

    int batchSize = 20000;
    this.ids = new String[files.size()];
    
    this.sim = DMath.createMatrix(files.size(), this.topics.size());
    DMatrix topicsNorm = this.topicsMat.vectorNorm();

    for(int f=0; f< files.size(); ) {
      if(f+batchSize > files.size())
        batchSize = files.size() - f;
      DMatrix batch = DMath.createMatrix(10000, this.dict.getSize());
      int[] sentLength = new int[batchSize];
      int countLines = 0;
      for(int j=0; j< batchSize; j++) {
        if(j%1000==0)
          System.out.printf("Processed files = %d\n", j);
        FIREDoc doc = parser.parse(files.get(f+j));
        
        // decide here what text goes in
        String title = doc.getTitle();
        if(this.useBoWDoc) {
          title = doc.getTitle()+ " " + doc.getContent();
        }
        
        this.ids[f+j] = StringUtils.removeExt(StringUtils.fileName(files.get(f+j)));
        int docLength = 0;
        if(this.useTitle || this.useBoWDoc) {
          DMatrix sMat = this.getRepresentation(ch, title);
          if(sMat != null) {
            if(countLines<batch.rows())
              batch.fillRow(countLines, sMat);
            else {
              batch.inflateRows(batch.rows()+10000, batch.columns());
              batch.fillRow(countLines, sMat);
            }
            docLength++;
            countLines++;
          }
        }
        if(this.useContent) {
          String[] sents = doc.getSentences(splitter);
          for(String sent: sents) {
            DMatrix sMat = this.getRepresentation(ch, sent);
            if(sMat != null) {
//              System.out.printf("Ok adding");
              if(countLines<batch.rows())
                batch.fillRow(countLines, sMat);
              else {
                batch.inflateRows(batch.rows()+10000, batch.columns());
                batch.fillRow(countLines, sMat);
              }
              docLength++;
              countLines++;
            }
          }
        }


        if(docLength==0) {
          DMatrix sMat = DMath.createZerosMatrix(1, this.dict.getSize());
          if(countLines<batch.rows())
            batch.fillRow(countLines, sMat);
          else {
            batch.inflateRows(batch.rows()+3000, batch.columns());
            batch.fillRow(countLines, sMat);
          }
          docLength++;
          countLines++;
//          System.out.printf("Its an issue here: DocLength = 0\n");
        }
        sentLength[j] = docLength;
//        System.out.printf("Doc %s has %d lines\n", ids[f+j], docLength);
      }

      int totLines = 0;
      for(int x=0; x<sentLength.length; x++)
        totLines+=sentLength[x];



      if(batch.rows()>countLines) {
        batch.truncateRows(countLines, batch.columns());
      }
      System.out.printf("Processing next batch (%d) from %d, totally it should be of size %d\n", batch.rows(), f, countLines);
      System.out.printf("Total lines should be %d while in batch there are %d lines.\n", totLines, batch.rows());
      // Process batch
      DMatrix proj = this.model.project(batch);
      
      DMatrix indexNorm = proj.vectorNorm();
      DMatrix batchSentSim = indexNorm.mmul(false, true, topicsNorm);
//      batchSentSim.print("Batch Sent sim");
      DMatrix batchSim = DMath.createMatrix(batchSize, batchSentSim.columns());

      int start = 0;
      for(int k=0; k<batchSize; k++) {
        batchSim.fillRow(k, (batchSentSim.sumRows(start, sentLength[k])).muli(1.0/(double)sentLength[k]));
//        batchSim.fillRow(k, (batchSentSim.sumRows(start, sentLength[k])));
        start+= sentLength[k];
      }
//      batchSim.print("Batch Sim");
      this.sim.fillMatrix(f, batchSim);
      f += batchSize;
    }
  }

/*  public DMatrix projectData(DMatrix v) {
    DMatrix h = v.mmul(false, false, this.vocab);
    return h;
  }*/

/*  // prepare index
  public void prepareIndex() {
    this.index = DMath.createMatrix(this.corpus.getSize(), this.model.outSize());
    for(int i=0; i<this.corpus.getSize(); i++) {
      Sentence s = this.corpus.get(i);
//      System.out.println(s.toString());
      DMatrix row = DMath.createMatrix(1, this.model.outSize());
      for(int j=0; j<s.getSize(); j++) {
//        System.out.println("Term = " + s.get(j));
//        this.vocab.getRow(s.get(j)).print();
        // TODO: This step might be huge overhead by using GPU
        row.addi(this.vocab.getRow(s.get(j)));
      }
      this.index.fillRow(i, row);
    }
  }*/

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
		ch.setup(TokenType.WORD, this.queryLang, path_to_terrier, pipeline);
    
    int i=0;
    for(Topic t: this.topics) {
      DMatrix row = DMath.createMatrix(1, this.model.outSize());
      String text = t.get(Tag.TITLE);
//      text += " "+t.get(Tag.DESC);
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
/*  // load queries
  public void loadQueries(String file) throws IOException {
    this.topics = FIRE.parseTopicFile(file);
    this.topicsMat = DMath.createMatrix(this.topics.size(), this.model.outSize());
    
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, this.queryLang, path_to_terrier, pipeline);
    
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
        double v = (Math.log(1.0+(double)docTf.get(k))/Math.log(2.0))*this.idf[k];
        topicRep.put(i, k, v);
      }
      
      
      for(String tok: tokens) {
        if(this.dict.getId(tok)>0) {
          // TODO: Might not be so efficient with GPU
          row.addi(this.vocab.getRow(this.dict.getId(tok)));
        }
      }
      this.topicsMat.fillRow(i, row);
      i++;
    }
  }*/
  
/*  // retrieve
  public void similarity() {
    DMatrix topicsNorm = this.topicsMat.vectorNorm();
    DMatrix indexNorm = this.index.vectorNorm();
    this.sim = topicsNorm.mmul(false, true, indexNorm);
  }*/

  public void printRankList(String outFile, int N) throws FileNotFoundException{
    PrintWriter p = new PrintWriter(outFile);
    for(int c=0; c<this.sim.columns(); c++) {
      double[] scores = new double[this.sim.rows()];
      for(int i=0; i<this.sim.rows(); i++) {
        scores[i] = this.sim.get(i, c);
        if(Double.isNaN(scores[i]))
          scores[i]=0.0;
      }
/*      System.out.printf("Generating RL for query %d and document list of size %d\n", c, scores.length);
      if(c==28) {
        for(int x=0; x<scores.length; x++)
          System.out.printf("%.4f ", scores[x]);
      }*/
      int[] rl = RankList.rankList(scores, N);
      String qid = this.topics.get(c).getID();
      for(int j=0; j<rl.length; j++) {
        p.println(qid+" Q0 "+this.ids[rl[j]]+" "+(j+1)+" "+ scores[rl[j]]);
      }
      
    }
    p.close();
    // get the ranklist of the docid sent and then print it in the TREC format taking help from Corpus
  }

  // evaluate
  public static void main(String[] args) throws IOException {
    LinearContinuousRetrieval ret = new LinearContinuousRetrieval();
    ret.docType = NEWSDocType.NAVBHARAT;

    ret.queryLang = Language.EN;
    ret.docLang = Language.HI;

    String corpus = "fire-new";

    ret.useTitle = true;
    ret.useContent = false;
    ret.useBoWDoc = false;

/*    String qrelFile = "sample/sample-qrel.txt";
    ret.loadQrel(qrelFile);
    System.out.printf("Qrel Loaded.\n");*/

    String modelPrefix = "OPCA-0.05";

    String dictFile = "data/"+corpus+"/joint/OPCA_dict.txt";
    ret.loadDict(dictFile);
    System.out.printf("Dictionary Loaded.\n");

    String modelFile = "data/"+corpus+"/joint/ProjMat-OPCA-0.05.mat";
    ret.loadModel(new File(modelFile));
    System.out.printf("Model Loaded.\n");

    ret.loadIDF("data/"+corpus+"/joint/CL-LSI-idf.txt");

    ret.loadQueries("sample/fire/topics/en.topics.126-175.2011.txt");
    System.out.printf("Topics Loaded.\n");
//    ret.topicsMat.print();

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
    ret.calculateSimilarity("/home/parth/workspace/data/fire/hi.docs.2011/docs/");
    System.out.printf("Collection Loaded.\n");

//    ret.prepareIndex();
//    System.out.printf("Index Prepared.\n");
//    ret.index.print();


//    ret.similarity();
//    System.out.printf("Similarity Estimated.\n");

    ret.printRankList("output/rl-continuous-"+modelPrefix+"-T.txt",1000);
//    ret.sim.print();

  }
}
