package retrieval;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.FileNotFoundException;
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

public class DNNContinuousRetrieval {
  Language qLang, dLang;
  Qrel qrel;
  Model dModel;
  Model qModel;
  List<Topic> topics;
  DMatrix qVocab, dVocab;
  Corpus corpus;
  DMatrix index;
  DMatrix topicsMat;
  DMatrix sim;
  String[] ids;
  FIREParserInterface parser;
  NEWSDocType docType;
  boolean useTitle = true, useContent = false;
  Map<String, Double> idf;
  List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
  
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
  
  // load both models
  public void loadModel(String qModelFile, Dictionary qDict, String dModelFile, Dictionary dDict) throws IOException {
    this.qModel = new AddModel();
    this.qModel.load(qModelFile, qDict);

    this.dModel = new AddModel();
    this.dModel.load(dModelFile, dDict);
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
  
  // project vocab
  public void projectVocabulary() {
  //  this.qVocab = this.qModel.projectVocabulary(5000);
  //  this.dVocab = this.dModel.projectVocabulary(5000);
    this.qVocab = this.qModel.projectVocabularyWithIdf(5000);
    this.dVocab = this.dModel.projectVocabularyWithIdf(5000);
  }

/*  // load collection
  public void loadCollection(String file) throws IOException {
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
		this.corpus = new Corpus();
    this.corpus.load(file, true, ch, this.model.dict(), false);
  }

  // prepare index
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

  public DMatrix getRepresentation(Channel ch, String text) {
    text = ch.getTokeniser().parse(text);
    text = ch.getTokeniser().clean(text);
    String[] tokens = text.split("_");
    DMatrix dVec = DMath.createZerosMatrix(1, this.dModel.outSize());
    boolean use = false;
    
      
    for(String tok: tokens) {
      if(this.dModel.dict().contains(tok)) {
        use = true;
        dVec.addi(this.dVocab.getRow(this.dModel.dict().getId(tok)));
      }
    }
    if(use)
      return dVec;
    else
      return null;
  }
  
  public void calculateSimilarity(String dir) throws IOException {
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";

		Channel ch = new SentFile("");
		this.pipeline.remove(PreProcessTerm.SW_REMOVAL);
		this.pipeline.remove(PreProcessTerm.STEM);
		ch.setup(TokenType.WORD, this.dLang, path_to_terrier, this.pipeline);

    this.parser = FIREParserFactory.getParser(this.docType);
    
    Splitter splitter = SplitterFactory.getSplitter(this.dLang);
    List<String> files = FileIO.getFilesRecursively(new File(dir), ".txt");

    int batchSize = 20000;
    this.ids = new String[files.size()];
    
    this.sim = DMath.createMatrix(files.size(), this.topics.size());
    DMatrix topicsNorm = this.topicsMat.vectorNorm();

    for(int f=0; f< files.size(); ) {
      if(f+batchSize > files.size())
        batchSize = files.size() - f;
      DMatrix batch = DMath.createMatrix(10000, this.dModel.outSize());
      int[] sentLength = new int[batchSize];
      int countLines = 0;
      for(int j=0; j< batchSize; j++) {
        if(j%1000==0)
          System.out.printf("Processed files = %d\n", j);
        FIREDoc doc = parser.parse(files.get(f+j));
        String title = doc.getTitle();
//        if(this.useContent)
          title += " " + doc.getContent();
        String[] sents = doc.getSentences(splitter);
        this.ids[f+j] = StringUtils.removeExt(StringUtils.fileName(files.get(f+j)));
        int docLength = 0;
        if(this.useTitle) {
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
      
      DMatrix indexNorm = batch.vectorNorm();
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

  // load queries
  public void loadQueries(String file) throws IOException {
    this.topics = FIRE.parseTopicFile(file);
    this.topicsMat = DMath.createMatrix(this.topics.size(), this.qModel.outSize());
    
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, this.qLang, path_to_terrier, this.pipeline);
    
    int i=0;
    for(Topic t: this.topics) {
      DMatrix row = DMath.createMatrix(1, this.qModel.outSize());
      String text = t.get(Tag.TITLE);
      text += " " + t.get(Tag.DESC);
      System.out.printf("query %s = %s\n", t.getID(), text);
      text = ch.getTokeniser().parse(text);
//      System.out.printf("query %s = %s\n", t.getID(), text);
      text = ch.getTokeniser().clean(text);
      System.out.printf("query %s = %s\n\n", t.getID(), text);
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
    DMatrix indexNorm = this.index.vectorNorm();
    this.sim = topicsNorm.mmul(false, true, indexNorm);
  }

/*  public void printRankList(String outFile, int N) throws FileNotFoundException{
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
  //
  public static void main(String[] args) throws IOException {
    DNNContinuousRetrieval ret = new DNNContinuousRetrieval();
/*    String qrelFile = "sample/sample-qrel.txt";
    ret.loadQrel(qrelFile);
    System.out.printf("Qrel Loaded.\n");*/

    ret.qLang = Language.EN;
    ret.dLang = Language.HI;

		ret.pipeline = new ArrayList<PreProcessTerm>();
		ret.pipeline.add(PreProcessTerm.SW_REMOVAL);
		ret.pipeline.add(PreProcessTerm.STEM);
    
    String corpus = "fire-new"; // fire-new or clef

    ret.docType = NEWSDocType.NAVBHARAT;
//    ret.docType = NEWSDocType.CLEF_FIRE;

    ret.useTitle = true;
    ret.useContent = false;

    String modelPrefix = "en-hi-title-10k-idf-new-content";
//    Dictionary dict = ret.loadDict("obj/"+modelPrefix+"/dict.txt");
//    Dictionary dDict = ret.loadDict("data/"+corpus+"/"+ret.dLang.getCode()+"/dict-400.txt");
//    Dictionary qDict = ret.loadDict("data/"+corpus+"/"+ret.qLang.getCode()+"/dict-100.txt");
    Dictionary dDict = ret.loadDict("data/"+corpus+"/"+ret.dLang.getCode()+"/dict-top10000.txt");
    Dictionary qDict = ret.loadDict("data/"+corpus+"/"+ret.qLang.getCode()+"/dict-top10000.txt");
    System.out.printf("Dictionary Loaded.\n");

    int iter = 64;
//    int iter = 65;
//    String qModelFile = "obj/tanh-cl-w-0.1-b-100-h-128/model_iter78.txt";
//    String dModelFile = "obj/tanh-hi-dict-400-b-100-h-128-new/model_iter20.txt";

    String qModelFile = "obj/tanh-fire-new-en-hi-cl-w-0.1-10k-b-100-h-128-bias-1.5/model_iter64.txt";
    String dModelFile = "obj/tanh-fire-hi-w-0.5-10k-b-100-h-128-bias-1.0/model_iter5.txt";
    
//    String qModelFile = "obj/tanh-clef-es-en-cl-w-0.1-10k-b-100-h-128-bias-2.0/model_iter35.txt";
//    String dModelFile = "obj/tanh-clef-en-w-0.5-10k-b-100-h-128-bias-1.0/model_iter8.txt";
    
//    String qModelFile = "obj/tanh-clef-es-en-cl-w-0.1-10k-b-100-h-128-bias-2.0/model_iter65.txt";
//    String dModelFile = "obj/tanh-clef-en-w-0.5-10k-b-100-h-128-bias-1.0-new/model_iter6.txt";
    
    ret.loadModel(qModelFile, qDict, dModelFile, dDict);
    System.out.printf("Model Loaded.\n");

    ret.loadIdf("data/"+corpus+"/joint/joint-idf.txt");
    ret.updateDictIdf();
    
    ret.projectVocabulary();
    System.out.printf("Vocabulary Projected.\n");

    ret.loadQueries("/home/parth/workspace/data/fire/topics/en.topics.126-175.2011.txt");
//    ret.loadQueries("/home/parth/workspace/jDNN/data/clef/ad-hoc/Top-es04-new.txt");
//    ret.loadQueries("/home/parth/workspace/jDNN/data/clef/ad-hoc/ESTopicsC301-C350.xml");
    System.out.printf("Topics Loaded.\n");
/*    DMatrix term1 = DMath.createMatrix(1, dict.getSize());
    term1.put(2759, 1.0);


    DMatrix term2 = DMath.createMatrix(1, dict.getSize());
    term2.put(2963, 1.0);

    ret.model.fProp(term1).print("Ideal 2759");
    ret.model.fProp(term2).print("Ideal 2963");

    ret.vocab.getRow(2759).print("with projected vocab: 2957");
    ret.vocab.getRow(2963).print("with projected vocab: 2963");*/
/*    ret.loadCollection("etc/data/fire/hi/title-only-with-did.txt");
    System.out.printf("Collection Loaded.\n");

    ret.prepareIndex();
    System.out.printf("Index Prepared.\n");*/
//    ret.index.print();

//    ret.topicsMat.print();

//    ret.similarity();
    ret.calculateSimilarity("/home/parth/workspace/data/fire/hi.docs.2011/docs/");
//    ret.calculateSimilarity("/home/parth/workspace/data/clef-data-jdnn/en-fire-format/gh-latimes94/");
    System.out.printf("Similarity Estimated.\n");

    ret.printRankList("output/rl-cl-dnn-"+modelPrefix+"-iter-"+iter+".txt",1000);
//    ret.sim.print();

  }
}
