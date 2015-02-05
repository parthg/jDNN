package models;

import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Arrays;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.Scanner;

import nn.Layer;
import nn.TanhLayer;
import common.Sentence;
import common.Dictionary;

import math.DMath;
import math.DMatrix;

public abstract class Model {
  Dictionary dict;
  List<Layer> layers;
 
  int thetaSize; 
  int inSize;
  int outSize;

  public Model() {
    this.layers = new ArrayList<Layer>();
    this.inSize = 0;
    this.outSize = 0;
    this.thetaSize = 0;
  }
  public void setDict(Dictionary _dict) {
    this.dict = _dict;
    this.inSize = this.dict.getSize();
  }

  public int outSize() {
    return outSize;
  }
  public Dictionary dict() {
    return this.dict;
  }

  public void addHiddenLayer(Layer l) {
    if(this.dict == null) {
      System.err.printf("[error] First set the Dictionary.. Exiting.\n");
      System.exit(0);
    }
    this.layers.add(l);
    this.outSize = l.getSize();
  }

  public void printArchitecture() {
    System.out.printf("Model architecture = %d ", this.inSize);

    for(int i=0; i<this.layers.size(); i++)
      System.out.printf("%d ", this.layers.get(i).getSize());

    System.out.printf(" Total number of parameters = %d\n", this.thetaSize);

  }

  public int getThetaSize() {
    return this.thetaSize;
  }

  public int getNumLayers() {
    return this.layers.size();
  }

  public void init() {
    init(1.0, 0.0);
  }
  
  public void init(double wScale, double bScale) {
    int tempInSize = this.inSize;
    for(Layer l: this.layers) {
      l.init(true, tempInSize, l.getSize(), wScale, bScale);
      tempInSize = l.getSize();
      this.thetaSize += l.getThetaSize();
    }
  }
  
  public void setParameters(double[] params) {
    assert (params.length == this.thetaSize):System.out.printf("The parameters length (%d) is not equal to that of model (%d)\n", params.length, this.thetaSize);
    int start = 0;
    Iterator<Layer> layerIt = this.layers.iterator();
    while(layerIt.hasNext()) {
      Layer l = layerIt.next();
      int lParamSize = l.getThetaSize();
      l.setParameters(Arrays.copyOfRange(params, start, start+lParamSize));
      start += lParamSize;
    }
  }

  public double[] getParameters() {
//    System.out.printf("Total Parameters in Model = %d\n", this.getThetaSize());
    double[] params = new double[this.thetaSize];
    int start = 0;
    Iterator<Layer> layerIt = this.layers.iterator();
    while(layerIt.hasNext()) {
      Layer l = layerIt.next();
      double[] lParams = l.getParameters();

      System.arraycopy(lParams, 0, params, start, lParams.length);
      start = l.getThetaSize();
    }
    return params;
  }
  public void save(String modelFile) throws IOException {
    PrintWriter p = new PrintWriter(modelFile);
    p.printf("#numLayers=%d\n",this.layers.size());
    Iterator<Layer> layerIt = this.layers.iterator();
    int lId = 1;
    while(layerIt.hasNext()) {
      Layer l = layerIt.next();
      p.printf("#Layer%d=%d %d\n", lId, l.getInSize(), l.getSize());
      l.clearDevice();
    }
    p.printf("#params=");
    double[] params = this.getParameters();
    for(int i=0; i<params.length; i++) {
      p.printf("%f ", params[i]);
    }
    p.printf("\n");
    p.close();
  }

  public void load(String modelFile, Dictionary _dict) throws IOException {
    this.setDict(_dict);
/*    Layer l = new TanhLayer(128);
    this.addHiddenLayer(l);
    this.init();*/
    
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(modelFile)));
    String line = "";
    while((line = br.readLine())!=null) {
      if(line.startsWith("#numLayers")) {
        String[] cols = line.split("=");
        int nLayers = Integer.parseInt(cols[1].trim());
        for(int lId = 0; lId<nLayers; lId++) {
          String lDetails = br.readLine();
          String[] lCols = lDetails.split("=");
          String[] lSize = lCols[1].trim().split(" ");
          int lLength = Integer.parseInt(lSize[1].trim());
          Layer l = new TanhLayer(lLength);
          this.addHiddenLayer(l);
        }
      }
      else if(line.startsWith("#params")) {
        this.printArchitecture();
        this.init();
        double[] params = new double[this.thetaSize];
        String[] cols = line.split("=");
        Scanner sc = new Scanner(cols[1].trim());
        int i=0;
        while(sc.hasNextDouble()) {
          params[i] = sc.nextDouble();
          i++;
        }
        this.setParameters(params);
      }
    }
    br.close();
  }

  public DMatrix projectVocabulary(int batchSize) {
    DMatrix proj = DMath.createMatrix(this.dict.getSize(), this.outSize);
    for(int i=0; i<this.dict.getSize(); ) {
      int b = Math.min(this.dict.getSize()-i, batchSize); // basically for the last batch of smaller size
      DMatrix vBatch = DMath.createMatrix(b, this.dict.getSize());
     
      for(int j=0; j<b; j++) {
        DMatrix v = DMath.createMatrix(1, this.dict.getSize());
        v.put(i+j, 1.0);
        vBatch.fillRow(j, v);
//        vBatch.put(j, i+j, 1.0);
      }
      DMatrix hBatch = this.fProp(vBatch);
      proj.fillMatrix(i, hBatch);
      i=i+b;
    }

/*    DMatrix vBatch = DMath.createMatrix(2, this.dict.getSize());
    DMatrix v1 = DMath.createMatrix(1, this.dict.getSize());
    v1.put(2759, 1.0);
    DMatrix v2 = DMath.createMatrix(1, this.dict.getSize());
    v2.put(2963, 1.0);

    vBatch.fillRow(0, v1);
    vBatch.fillRow(1, v2);

    DMatrix hBatch = this.fProp(vBatch);
    hBatch.print("Projected 2759 and 2963 from projected vocab");
    proj.fillMatrix(2759, hBatch.getRow(0));
    proj.fillMatrix(2963, hBatch.getRow(1));*/
    return proj;
  }
  
  public void copyHtoD() {
    Iterator<Layer> layerIt = this.layers.iterator();
    while(layerIt.hasNext()) {
      Layer l = layerIt.next();
      l.copyHtoD();
    }
  }

  public void copyDtoH() {
    Iterator<Layer> layerIt = this.layers.iterator();
    while(layerIt.hasNext()) {
      Layer l = layerIt.next();
      l.copyDtoH();
    }
  }

  public void clearDevice() {
    Iterator<Layer> layerIt = this.layers.iterator();
    while(layerIt.hasNext()) {
      Layer l = layerIt.next();
      l.clearDevice();
    }
  }

  public abstract DMatrix fProp(Sentence input);
  public abstract DMatrix fProp(DMatrix input);
  public abstract DMatrix getRepresentation(DMatrix sentMatrix);
  public abstract DMatrix bProp(Sentence s1, Sentence s2);
  public abstract DMatrix bProp(Sentence s, DMatrix error);
  public abstract DMatrix bProp(DMatrix input, DMatrix error);
  public abstract DMatrix bProp(DMatrix input, DMatrix rep, DMatrix error);
}
