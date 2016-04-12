package models;

import nn.Layer;
import nn.SiameseLayer;
import math.DMath;
import math.DMatrix;
import common.Dictionary;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.Scanner;

import java.util.Iterator;
import java.util.ListIterator;
import java.util.Stack;

import common.Sentence;

public class S2Net extends Model {
  public S2Net() {
    super();
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
//          Layer l = new TanhLayer(lLength);
          Layer l = new SiameseLayer(lLength);
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
  
  public DMatrix fProp(Sentence sent) {
    DMatrix rep = DMath.createZerosMatrix(1,super.outSize);
    rep.copyHtoD();
    Iterator<Integer> sentIt = sent.words.iterator();
    while(sentIt.hasNext()) {
      DMatrix input = this.dict.getRepresentation(sentIt.next());
      Iterator<Layer> layerIt = this.layers.iterator();
      DMatrix temp = input;
      while(layerIt.hasNext()) {
        Layer l = layerIt.next();
        temp = l.fProp(temp);
      }
      rep.addi(temp);
    }
    rep.copyDtoH();
    rep.close();
    return rep;
  } 

  public DMatrix getRepresentation(DMatrix sentMatrix) {
    DMatrix temp = this.layers.get(0).fProp(sentMatrix);

    if(this.getNumLayers()>1) {
      for(int i=1; i<getNumLayers(); i++) {
        temp = this.layers.get(i).fProp(temp);
      }
    }
    return temp.sumRows();
  }

  public DMatrix fProp(DMatrix input) {
    DMatrix temp = this.layers.get(0).fProp(input);
    if(this.getNumLayers()>1) {
      for(int i=1; i<getNumLayers(); i++) {
        temp = this.layers.get(i).fProp(temp);
      }
    }
    return temp;
  }
  
  public DMatrix bProp(Sentence s1, Sentence s2) {
    // send s2 up -- use it as label
    // send s1 up 
    // calculate error (1-2)
    // backpropagate error -- for each word!
    
    // TODO: initialize it with persist = true so that it is not copied to GPU in each iterations.
    DMatrix grads = DMath.createZerosMatrix(1, this.thetaSize);
    grads.copyHtoD(); 
    DMatrix label = this.fProp(s2);
    DMatrix pred = this.fProp(s1);

    DMatrix error = pred.sub(label);
    Iterator<Integer> sentIt = s1.words.iterator();
    // for each word
    while(sentIt.hasNext()) {
      int start = 0;
      double[] tempGrads = new double[this.thetaSize];
      DMatrix word = this.dict.getRepresentation(sentIt.next());
      Iterator<Layer> layerIt = this.layers.iterator();
      DMatrix rep = word;
      Stack<DMatrix> input = new Stack<DMatrix>();
      input.push(word);

      // fprop and store the representations at each layer in stack
      while(layerIt.hasNext()) {
        Layer l = layerIt.next();
        rep = l.fProp(rep);
        input.push(rep);
      }
      ListIterator<Layer> layerRevIt = this.layers.listIterator(this.layers.size());
      DMatrix tempError = error;
      rep = input.pop();
      // backprop it
      while(layerRevIt.hasPrevious()) {
        Layer l = layerRevIt.previous();
        DMatrix lGrads = l.bProp(rep, tempError);
        double[] lParamGrads = l.getParamGradients(input.peek(), lGrads);
        // TODO: Make it conditioned if there is next layer. otherwise its unnecessary
        tempError = lGrads.mmul(false, true, l.getWeights());
        rep = input.pop();
        System.arraycopy(lParamGrads, 0, tempGrads, start, lParamGrads.length);
        start = l.getThetaSize();
      }
      grads.addi(DMath.createMatrix(1, this.thetaSize, tempGrads));
    }

    grads.copyDtoH();
    grads.close();
    // TODO:before returning close it?
    
    return grads;
  }
  public DMatrix bProp(Sentence s1, DMatrix error) {
    DMatrix grads = DMath.createZerosMatrix(1, this.thetaSize);
    grads.copyHtoD();
    Iterator<Integer> sentIt = s1.words.iterator();
    // for each word
    while(sentIt.hasNext()) {
      int start = 0;
      double[] tempGrads = new double[this.thetaSize];
      DMatrix word = this.dict.getRepresentation(sentIt.next());
      Iterator<Layer> layerIt = this.layers.iterator();
      DMatrix rep = word;
      Stack<DMatrix> input = new Stack<DMatrix>();
      input.push(word);

      // fprop and store the representations at each layer in stack
      while(layerIt.hasNext()) {
        Layer l = layerIt.next();
        rep = l.fProp(rep);
        input.push(rep);
      }
      ListIterator<Layer> layerRevIt = this.layers.listIterator(this.layers.size());
      DMatrix tempError = error;
      rep = input.pop();
      // backprop it
      int layersLeft = this.getNumLayers();
      while(layerRevIt.hasPrevious()) {
        layersLeft--;
        Layer l = layerRevIt.previous();
        DMatrix lGrads = l.bProp(rep, tempError);
        double[] lParamGrads = l.getParamGradients(input.peek(), lGrads);
        // Make it conditioned if there is next layer. otherwise its unnecessary
        if(layersLeft>0)
          tempError = lGrads.mmul(false, true, l.getWeights());
        rep = input.pop();
        System.arraycopy(lParamGrads, 0, tempGrads, start, lParamGrads.length);
        start = l.getThetaSize();
      }
      grads.addi(DMath.createMatrix(1, this.thetaSize, tempGrads));
    }
    grads.copyDtoH();
    grads.close();
    return grads;
  }
  
  /* input  = matrix
   * rep    = matrix
   * error  = matrix
   */
  public DMatrix bProp(DMatrix input, DMatrix rep, DMatrix error) {
    
    DMatrix grads = DMath.createZerosMatrix(1, this.thetaSize);
//    DMatrix lGrads = this.layers.get(0).bProp(rep, error);
    double[] lParamGrads = this.layers.get(0).getParamGradients(input, error);
    
    System.arraycopy(lParamGrads, 0, grads.data(), 0 , lParamGrads.length);

    return grads;
  }
  
  public DMatrix bProp(DMatrix input, DMatrix error) {

    // layerwise fprop -- get rep and save intermediate rep
    // layerwise bprop 
    
    DMatrix grads = DMath.createZerosMatrix(1, this.thetaSize);
    DMatrix rep = this.fProp(input);

    int start = this.thetaSize;
    if(this.getNumLayers()>1) {
      for(int i=getNumLayers()-1; i>0; i--) {
        DMatrix lGrads = this.layers.get(i).bProp(rep, error);
//        lGrads.printDim("lGrads");
        DMatrix interInput = this.layers.get(i-1).getData();
//        interInput.printDim("interInput");
        
        // TODO: for the deep model, input will  be intermediate representation
        double[] lParamGrads = this.layers.get(i).getParamGradients(interInput, lGrads);
        System.arraycopy(lParamGrads, 0, grads.data(), start-this.layers.get(i).getThetaSize(), lParamGrads.length);

        error = lGrads.mmul(false, true, this.layers.get(i).getWeights());
        rep = this.layers.get(i-1).getData();
        start = start - this.layers.get(i).getThetaSize();
      }
    }
    // outermost layer
    DMatrix lGrads = this.layers.get(0).bProp(rep, error);
    
    double[] lParamGrads = this.layers.get(0).getParamGradients(input, lGrads);
    System.arraycopy(lParamGrads, 0, grads.data(), start-this.layers.get(0).getThetaSize(), lParamGrads.length);
    
    return grads;
  }
}
