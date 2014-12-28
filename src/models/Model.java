package models;

import java.util.List;
import java.util.ArrayList;
import org.jblas.DoubleMatrix;

import nn.Layer;
import common.Sentence;
import common.Dictionary;

public abstract class Model {
  Dictionary dict;
  List<Layer> layers;
  
  int inSize;
  int outSize;

  public Model() {
    this.layers = new ArrayList<Layer>();
    this.inSize = 0;
    this.outSize = 0;
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

  public void init() {
    int tempInSize = this.inSize;
    for(Layer l: this.layers) {
      l.init(true, tempInSize, l.getSize());
      tempInSize = l.getSize();
    }
  }
  public double[] getParameters() {
    double[] params = new double[0];
    Iterator<Layer> layerIt = this.layers.iterator();
    DoubleMatrix temp = input;
    while(layerIt.hasNext()) {
      Layer l = layerIt.next();
      double[] wParams = l.getWeights().toArray();
      double[] bParams = l.getBiases().toArray();

      double[] lParams= new double[wParams.length+bParams.length];
      System.arraycopy(wParams, 0, lParams, 0, wParams.length);
      System.arraycopy(bParams, 0, lParams, wParams.length, bParams.length);

      System.arraycopy(lParams, 0, params, params.length, lParams.length);
    }
    return params;
  }
  public abstract DoubleMatrix output(Sentence input);
  
}
