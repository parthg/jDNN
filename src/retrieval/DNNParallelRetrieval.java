package retrieval;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.File;
import java.io.IOException;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import math.DMath;
import math.DMatrix;
import models.Model;
import models.AddModel;
import models.CLTrainModel;
import models.CanonicalCorrelation;
import common.Dictionary;
import common.Metric;

import es.upv.nlel.utils.Language;

public class DNNParallelRetrieval {
  Model qModel, dModel;
  DMatrix qVocab, dVocab;

  DMatrix enData, hiData;
  DMatrix enProj, hiProj;
  CanonicalCorrelation cca;
  Map<String, Double> idf;

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

/*    DMatrix idfMaskQ = DMath.createMatrix(this.qVocab.rows(), this.qVocab.cols());
    DMatrix idfMaskD = DMath.createMatrix(this.dVocab.rows(), this.dVocab.cols());

    for(int i=0; i<idfMaskQ.rows(); i++) {
      idfMaskQ.fillRow(this.qModel.dict.getIdf(i));
    }*/

  }

  // print dict embeddings
  public void printEmbeddings(String file1, String file2) throws IOException {
    this.qVocab.printToFile(new File(file1));
    this.dVocab.printToFile(new File(file2));
  }
  
  // Loads the data for both languages from a sparse matrix respresntation file, 
  // projects it and stores in the class objects
  public void loadData(String enFile, String hiFile) throws IOException {
    this.enProj = DMath.createMatrix(20000, this.qModel.outSize());
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(enFile)));
    String line = "";
    int count = 0;
    while((line = br.readLine())!=null) {
      if(this.enProj.rows()<=count) {
        this.enProj.inflateRows(this.enProj.rows()+5000, this.enProj.columns());
      }
      DMatrix row = DMath.createZerosMatrix(1, this.qModel.outSize());
      String[] cols = line.split(" ");
/*      List<String> tokens = new ArrayList<String>();
      for(int k=0; k<cols.length; k++) {
        tokens.add(cols[k]);
      }
      List<Integer> tids = this.qModel.dict().getTermIds(tokens);
      for(int tid: tids) {
        row.addi(this.qVocab.getRow(tid));
      }*/
      for(int k=0; k<cols.length; k++) {
        int t = Integer.parseInt(cols[k].trim());
        row.addi(this.qVocab.getRow(t));
//        row.addi(this.qVocab.getRow(t).mul(this.qModel.dict().getIdf(t)));
      }
      this.enProj.fillRow(count, row);
      count++;
    }
    br.close();
    this.enProj.truncateRows(count, this.enProj.columns());

    this.hiProj = DMath.createMatrix(20000, this.dModel.outSize());
    br = new BufferedReader(new InputStreamReader(new FileInputStream(hiFile)));
    line = "";
    count = 0;
    while((line = br.readLine())!=null) {
      if(this.hiProj.rows()<=count) {
        this.hiProj.inflateRows(this.hiProj.rows()+5000, this.hiProj.columns());
      }
      DMatrix row = DMath.createZerosMatrix(1, this.dModel.outSize());
      String[] cols = line.split(" ");
/*      List<String> tokens = new ArrayList<String>();
      for(int k=0; k<cols.length; k++) {
        tokens.add(cols[k]);
      }
      List<Integer> tids = this.dModel.dict().getTermIds(tokens);
      for(int tid: tids) {
        row.addi(this.dVocab.getRow(tid));
      }*/
      for(int k=0; k<cols.length; k++) {
        int t = Integer.parseInt(cols[k].trim());
        row.addi(this.dVocab.getRow(t));
//        row.addi(this.dVocab.getRow(t).mul(this.dModel.dict().getIdf(t)));
      }
      this.hiProj.fillRow(count, row);
      count++;
    }
    br.close();
    this.hiProj.truncateRows(count, this.hiProj.columns());
  }
  
  public void loadCCA(String xCoefFile, String yCoefFile, String yCenterFile) throws IOException {
    this.cca = new CanonicalCorrelation();
		cca.loadXCoeff(xCoefFile);
		cca.loadYCoeff(yCoefFile);
		cca.loadYCenter(yCoefFile);
  }

  public DMatrix loadData(int dim, String file) throws IOException {
    DMatrix m = DMath.createMatrix(10000, dim);
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
    String line = "";
    int i=0;
    while((line = br.readLine())!=null) {
      if(i>=m.rows())
        m.inflateRows(m.rows()+5000, m.columns());
      String[] cols = line.split(" ");
      for(int j=0; j<cols.length; j++)
        m.put(i, j, Double.parseDouble(cols[j].trim()));
      i++;
    }
    if(m.rows()>i)
      m.truncateRows(i, m.columns());
    return m;

  }

  public void canCorTranslate() {
    this.enProj = this.cca.getXRep(this.enData);
    this.hiProj = this.cca.getYRep(this.hiData);
  }

/*  // load dict
  public void loadDict(String file) throws IOException {
    this.dict = new Dictionary();
    this.dict.load(file);
  }
  
  // load model
  public void loadModel(File file) throws IOException {
    this.model = new LinearModel(this.dict.getSize(), 128);
    this.model.load(file);
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

  public void loadData(String enFile, String hiFile) throws IOException {
    this.enData = DMath.createMatrix(20000, this.dict.getSize());
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(enFile)));
    String line = "";
    int count = 0;
    while((line = br.readLine())!=null) {
      if(this.enData.rows()<=count) {
        this.enData.inflateRows(this.enData.rows()+5000, this.enData.columns());
      }
      String[] cols = line.split(" ");
      Map<Integer, Integer> docTf = new HashMap<Integer, Integer>();
      for(int k=0; k<cols.length; k++) {
        int tid = Integer.parseInt(cols[k]);
        if(docTf.containsKey(tid))
          docTf.put(tid, docTf.get(tid)+1);
        else
          docTf.put(tid, 1);
      }
      for(int k : docTf.keySet()) {
        double v = (Math.log(1.0+(double)docTf.get(k))/Math.log(2.0))*idf[k];
        this.enData.put(count, k, v);
      }
      count++;
    }
    br.close();
    this.enData.truncateRows(count, this.enData.columns());

    this.enProj = this.model.project(this.enData);
    this.enData = DMath.createMatrix(1, 1);
    
    
    this.hiData = DMath.createMatrix(20000, this.dict.getSize());
    br = new BufferedReader(new InputStreamReader(new FileInputStream(hiFile)));
    line = "";
    count = 0;
    while((line = br.readLine())!=null) {
      if(this.hiData.rows()<=count) {
        this.hiData.inflateRows(this.hiData.rows()+5000, this.hiData.columns());
      }
      String[] cols = line.split(" ");
      Map<Integer, Integer> docTf = new HashMap<Integer, Integer>();
      for(int k=0; k<cols.length; k++) {
        int tid = Integer.parseInt(cols[k]);
        if(docTf.containsKey(tid))
          docTf.put(tid, docTf.get(tid)+1);
        else
          docTf.put(tid, 1);
      }
      for(int k : docTf.keySet()) {
        double v = (Math.log(1.0+(double)docTf.get(k))/Math.log(2.0))*idf[k];
        this.hiData.put(count, k, v);
      }
      count++;
    }
    br.close();
    this.hiData.truncateRows(count, this.hiData.columns());
    this.hiProj = this.model.project(this.hiData);
    this.hiData = DMath.createMatrix(1, 1);
  }*/

  public double mrr() throws IOException {
    DMatrix enNorm = this.enProj.vectorNorm();
    DMatrix hiNorm = this.hiProj.vectorNorm();

    enNorm.printDim("enNorm");
    hiNorm.printDim("esNorm");
    
//    enNorm.print("enNormMatrixData");
//    DMatrix hiNorm = (CLTrainModel.loadMatrix(128, "data/fire/joint/DNN-subparallel-projected-hi-test-part.dat")).vectorNorm();

    DMatrix sim = enNorm.mmul(false, true, hiNorm);
    return Metric.mrr(sim);
  }

  public static void main(String[] args) throws IOException {
    DNNParallelRetrieval ret = new DNNParallelRetrieval();
/*  // for CCA  
    ret.enData = ret.loadData(128, "data/fire/joint/DNN-subparallel-projected-en-test.dat");
    ret.hiData = ret.loadData(128, "data/fire/joint/DNN-subparallel-projected-hi-test.dat");

    ret.loadCCA("data/fire/joint/enCoef-128-128", "data/fire/joint/hiCoef-128-128","data/fire/joint/hiCenter-1-128" );
    ret.canCorTranslate();
    System.out.printf("MRR = %.6f\n", ret.mrr());*/
   
    /***************
    //The working configurations:
    Dictionary qDict = ret.loadDict("data/fire/en/dict-100.txt");
    Dictionary dDict = ret.loadDict("data/fire/hi/dict-400.txt");
    System.out.printf("Dictionary Loaded.\n");

    String qModelFile = "obj/tanh-cl-w-0.1-b-100-h-300-128/model_iter37.txt";
    String dModelFile = "obj/tanh-hi-dict-400-b-100-h-128-new/model_iter20.txt";
    ***************/

  
    /*********************
    // Full Dict
    Dictionary qDict = ret.loadDict("data/fire/en/dict-parallel.txt");
    Dictionary dDict = ret.loadDict("data/fire/hi/dict-titles-full.txt");
    System.out.printf("Dictionary Loaded.\n");

    String qModelFile = "obj/tanh-cl-dict-40k-w-0.1-b-50-h-128/model_iter23.txt";
    String dModelFile = "obj/tanh-b-100-h-128/model_iter14.txt";
    **********************/

    Language lang1 = Language.DE; // query lang
    Language lang2 = Language.EN; // doc lang

    String corpus = "clef";
    String testSuffix = "-25k";
    String jointDir = "joint-de";

    Dictionary qDict = ret.loadDict("data/"+corpus+"/"+lang1.getCode()+"/dict-top10000.txt");
    Dictionary dDict = ret.loadDict("data/"+corpus+"/"+lang2.getCode()+"/dict-top10000.txt");
    System.out.printf("Dictionary Loaded.\n");

//    String qModelFile = "obj/tanh-clef-en-es-cl-w-0.5-10k-b-200-h-128/model_iter50.txt";
//    String qModelFile = "obj/tanh-clef-en-es-cl-w-0.1-10k-b-100-h-128-bias-2.0/model_iter33.txt";
//    String qModelFile = "obj/tanh-clef-en-es-cl-w-0.1-10k-b-100-h-1000-128-bias-1.0-deep/model_iter10.txt";
//    String dModelFile = "obj/tanh-es-dict-top-10k-b-200-h-128/model_iter29.txt";
    
//    String qModelFile = "obj/tanh-fire-new-en-hi-cl-w-0.1-10k-b-100-h-128-bias-1.5/model_iter64.txt";
//    String dModelFile = "obj/tanh-fire-hi-w-0.5-10k-b-100-h-128-bias-1.0/model_iter5.txt";

//    String qModelFile = "obj/tanh-clef-es-en-cl-w-0.1-10k-b-100-h-128-bias-2.0/model_iter35.txt";
//    String dModelFile = "obj/tanh-clef-en-w-0.5-10k-b-100-h-128-bias-1.0/model_iter8.txt";

//    String qModelFile = "obj/tanh-clef-es-en-cl-w-0.1-10k-b-100-h-128-bias-2.0/model_iter37.txt";
    String qModelFile = "obj/tanh-clef-de-en-cl-w-0.1-10k-b-100-h-128-bias-1.5-new/model_iter44.txt";
    String dModelFile = "obj/tanh-clef-en-w-0.5-10k-b-100-h-128-bias-1.0-new/model_iter6.txt";

    ret.loadModel(qModelFile, qDict, dModelFile, dDict);
    System.out.printf("Model Loaded.\n");

    ret.loadIdf("data/"+corpus+"/"+jointDir+"/joint-idf.txt");
    ret.updateDictIdf();
    
    ret.projectVocabulary();
    ret.printEmbeddings("data/clef/"+jointDir+"/"+lang1.getCode()+"-embeddings.dat", "data/clef/"+jointDir+"/"+lang2.getCode()+"-embeddings.dat");
    System.out.printf("Vocabulary Projected.\n");
    
//    ret.loadData("data/clef/joint/DNN-subparallel-en-test.dat", "data/clef/joint/DNN-subparallel-hi-test.dat");
//    ret.loadData("data/clef/joint/en-test-25k.dat", "data/clef/joint/hi-test-25k.dat");
//    ret.loadData("data/fire/joint-full/DNN-subparallel-en-test-21k.dat", "data/fire/joint-full/DNN-subparallel-hi-test-21k.dat");
    ret.loadData("data/"+corpus+"/"+jointDir+"/DNN-subparallel-"+lang1.getCode()+"-test"+testSuffix+".dat", "data/"+corpus+"/"+jointDir+"/DNN-subparallel-"+lang2.getCode()+"-test"+testSuffix+".dat");
    
    System.out.printf("Data Projected.\n");
    
    System.out.printf("MRR = %.6f\n", ret.mrr());
  }
}
