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
    DMatrix rep = DMath.createZerosMatrix(1,super.outSize);
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
    return rep;
  }

  public DMatrix bProp(Sentence s1, Sentence s2) {
    // send s2 up -- use it as label
    // send s1 up 
    // calculate error (1-2)
    // backpropagate error -- for each word!
    DMatrix grads = DMath.createZerosMatrix(1, this.thetaSize); 
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
        tempError = lGrads.mmul(l.getWeights().transpose());
        rep = input.pop();
        System.arraycopy(lParamGrads, 0, tempGrads, start, lParamGrads.length);
        start = l.getThetaSize();
      }
      grads.addi(DMath.createMatrix(1, this.thetaSize, tempGrads));
    }
    return grads;
  }
}
