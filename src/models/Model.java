package models;

import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Arrays;
//import org.jblas.DoubleMatrix;

import java.io.PrintWriter;
import java.io.IOException;

import nn.Layer;
import common.Sentence;
import common.Dictionary;

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

  public int getThetaSize() {
    return this.thetaSize;
  }

  public int getNumLayers() {
    return this.layers.size();
  }

  public void init() {
    int tempInSize = this.inSize;
    for(Layer l: this.layers) {
      l.init(true, tempInSize, l.getSize());
      tempInSize = l.getSize();
      this.thetaSize += l.getThetaSize();
    }
  }
  
  public void setParameters(double[] params) {
    int start = 0;
    Iterator<Layer> layerIt = this.layers.iterator();
    while(layerIt.hasNext()) {
      Layer l = layerIt.next();
      int lParamSize = l.getThetaSize();
      l.setParameters(Arrays.copyOfRange(params, start, lParamSize));
      start = lParamSize;
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
