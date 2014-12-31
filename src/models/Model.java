package models;

import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Arrays;
import org.jblas.DoubleMatrix;

import nn.Layer;
import common.Sentence;
import common.Dictionary;

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
    System.out.printf("Total Parameters in Model = %d\n", this.getThetaSize());
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
  public abstract DoubleMatrix fProp(Sentence input);
  public abstract DoubleMatrix bProp(Sentence s1, Sentence s2);
}
