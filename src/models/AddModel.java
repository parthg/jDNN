package models;

import nn.Layer;
import math.DMath;
import math.DMatrix;

//import org.jblas.DoubleMatrix;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Stack;

import common.Sentence;

public class AddModel extends Model {
  public AddModel() {
    super();
  }
  public DMatrix fProp(Sentence sent) {
    DMatrix rep = DMath.createZerosMatrix(1,super.outSize, true);
    Iterator<Integer> sentIt = sent.words.iterator();
//    System.out.printf("original vectors: %d", sent.getSize());
    while(sentIt.hasNext()) {
      DMatrix input = this.dict.getRepresentation(sentIt.next());
//      input.print();
      Iterator<Layer> layerIt = this.layers.iterator();
      DMatrix temp = input;
      while(layerIt.hasNext()) {
        Layer l = layerIt.next();
        temp = l.fProp(temp);
      }
//      temp.print();
      rep.addi(temp);
    }
    rep.copyDtoH();
    rep.close();
    return rep;
  }

/*  public DMatrix fProp(DMatrix input) {
    Iterator<Layer> layerIt = this.layers.iterator();
    DMatrix temp = input;
    while(layerIt.hasNext()) {
      Layer l = layerIt.next();
      temp = l.fProp(temp);
    }
    return temp.sumRows();
  }*/
 

  public DMatrix getRepresentation(DMatrix sentMatrix) {
    DMatrix temp = this.layers.get(0).fProp(sentMatrix);
    return temp.sumRows();
  }

  public DMatrix fProp(DMatrix input) {
//    System.out.printf("batch of %d\n", input.rows());
    DMatrix temp = this.layers.get(0).fProp(input);

//    temp.print();
    return temp;
  }
  
  public DMatrix bProp(Sentence s1, Sentence s2) {
    // send s2 up -- use it as label
    // send s1 up 
    // calculate error (1-2)
    // backpropagate error -- for each word!
    
    // TODO: initialize it with persist = true so that it is not copied to GPU in each iterations.
    DMatrix grads = DMath.createZerosMatrix(1, this.thetaSize, true); 
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

    // TODO:before returning close it?
    
    return grads;
  }
  public DMatrix bProp(Sentence s1, DMatrix error) {
    // send s2 up -- use it as label
    // send s1 up 
    // calculate error (1-2)
    // backpropagate error -- for each word!
    
    // TODO: initialize it with persist = true so that it is not copied to GPU in each iterations.
    DMatrix grads = DMath.createZerosMatrix(1, this.thetaSize, true); 
/*    DMatrix label = this.fProp(s2);
    DMatrix pred = this.fProp(s1);

    DMatrix error = pred.sub(label);*/
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
        // TODO: Make it conditioned if there is next layer. otherwise its unnecessary
        if(layersLeft>0)
          tempError = lGrads.mmul(false, true, l.getWeights());
        rep = input.pop();
        System.arraycopy(lParamGrads, 0, tempGrads, start, lParamGrads.length);
        start = l.getThetaSize();
      }
      grads.addi(DMath.createMatrix(1, this.thetaSize, tempGrads));
    }

    // TODO:before returning close it?
    
    return grads;
  }
  
  /* input  = matrix
   * rep    = matrix
   * error  = vector
   */
  public DMatrix bProp(DMatrix input, DMatrix rep, DMatrix error) {
    
    DMatrix grads = DMath.createZerosMatrix(1, this.thetaSize);

    DMatrix batchError = DMath.createMatrix(rep.rows(), rep.columns());
    batchError.fillWithArray(error);

    DMatrix lGrads = this.layers.get(0).bProp(rep, batchError);
    
//    DMatrix batchLGrads = DMath.createMatrix(input.rows(), lGrads.columns());
//    batchLGrads.fillWithArray(lGrads);
    double[] lParamGrads = this.layers.get(0).getParamGradients(input, lGrads);
    
    System.arraycopy(lParamGrads, 0, grads.data(), 0 , lParamGrads.length);

    return grads;
  }
  
  public DMatrix bProp(DMatrix input, DMatrix error) {

    // layerwise fprop -- get rep and save intermediate rep
    // layerwise bprop 
    
    DMatrix grads = DMath.createZerosMatrix(1, this.thetaSize);
    DMatrix rep = this.fProp(input);

    DMatrix lGrads = this.layers.get(0).bProp(rep, error);
    double[] lParamGrads = this.layers.get(0).getParamGradients(input, lGrads);
    System.arraycopy(lParamGrads, 0, grads.data(), 0, lParamGrads.length);

    return grads;
  }
}
