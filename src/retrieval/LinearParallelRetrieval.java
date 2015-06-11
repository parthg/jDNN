package retrieval;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.File;
import java.io.IOException;

import java.util.Map;
import java.util.HashMap;
import math.DMath;
import math.DMatrix;
import models.LinearModel;
import models.Model;
import models.S2Net;
import common.Dictionary;
import common.Metric;

/** This class provies facility to generate results of parallel sentence retrieval Task.
 *
 */
public class LinearParallelRetrieval {
  LinearModel model;
  Model nnModel;
  Dictionary dict;
  DMatrix enData, hiData;
  DMatrix enProj, hiProj;
  double[] idf;
  String modelType;


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

  public void loadS2Net(String file) throws IOException {
    this.nnModel = new S2Net();
    this.nnModel.load(file, this.dict);
  }
  
  // Loads idf data calculated on the train parallel data
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

  // Loads the data for both languages from a sparse matrix respresntation file, 
  // projects it and stores in the class objects
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

    if(this.modelType.equals("S2Net"))
      this.enProj = this.nnModel.fProp(this.enData);
    else
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
    if(this.modelType.equals("S2Net"))
      this.hiProj = this.nnModel.fProp(this.hiData);
    else
      this.hiProj = this.model.project(this.hiData);
    this.hiData = DMath.createMatrix(1, 1);
  }

  public void loadAEProjData(int dim, String enProjFile, String hiProjFile) throws IOException {
    this.enProj = models.CLTrainModel.loadMatrix(dim, enProjFile);
    this.hiProj = models.CLTrainModel.loadMatrix(dim, hiProjFile);
  }

  // calculates MRR for the projected parallel data
  public double mrr() {
    DMatrix enNorm = this.enProj.vectorNorm();
    DMatrix hiNorm = this.hiProj.vectorNorm();

    DMatrix sim = enNorm.mmul(false, true, hiNorm);
    return Metric.mrr(sim);
  }

  public static void main(String[] args) throws IOException {
    LinearParallelRetrieval ret = new LinearParallelRetrieval();
    ret.modelType = "AE";
    ret.loadDict("data/fire/joint/CL-LSI-dict.txt");
    ret.loadIDF("data/fire/joint/CL-LSI-idf.txt");
    if(ret.modelType.equals("CL-LSI") || ret.modelType.equals("OPCA")) {
      ret.loadModel(new File("data/fire/joint/ProjMat-OPCA-0.05.mat"));
      ret.loadData("data/fire/joint/joint-test-en.dat", "data/fire/joint/joint-test-hi.dat");
    }
    else if(ret.modelType.equals("AE")) {
      ret.loadAEProjData(128, "data/fire/joint/ae/projected-en.dat", "data/fire/joint/ae/projected-hi.dat");
    }
    else if(ret.modelType.equals("S2Net")) {
      ret.loadS2Net("obj/s2net-cl-h-128-new/model_iter6.txt");
      ret.loadData("data/fire/joint/joint-test-en.dat", "data/fire/joint/joint-test-hi.dat");
    }
    else {
      System.out.printf("Not a proper Model Type %s\nExiting.", ret.modelType);
      System.exit(0);
    }
    System.out.println("Data projected.\n");
    System.out.printf("MRR = %.6f\n", ret.mrr());
  }
}
