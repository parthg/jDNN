package models;

import nn.Layer;

import org.jblas.DoubleMatrix;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Stack;

import common.Sentence;

public class AddModel extends Model {
  public AddModel() {
    super();
  }
  public DoubleMatrix fProp(Sentence sent) {
    DoubleMatrix rep = DoubleMatrix.zeros(1,super.outSize);
    Iterator<Integer> sentIt = sent.words.iterator();
    while(sentIt.hasNext()) {
      DoubleMatrix input = this.dict.getRepresentation(sentIt.next());
      Iterator<Layer> layerIt = this.layers.iterator();
      DoubleMatrix temp = input;
      while(layerIt.hasNext()) {
        Layer l = layerIt.next();
        temp = l.fProp(temp);
      }
      rep.addi(temp);
    }
    return rep;
  }

  public DoubleMatrix bProp(Sentence s1, Sentence s2) {
    // send s2 up -- use it as label
    // send s1 up 
    // calculate error (1-2)
    // backpropagate error -- for each word!
    DoubleMatrix grads = DoubleMatrix.zeros(1, this.thetaSize); 
    DoubleMatrix label = this.fProp(s2);
    DoubleMatrix pred = this.fProp(s1);

    DoubleMatrix error = pred.sub(label);
    Iterator<Integer> sentIt = s1.words.iterator();
    // for each word
    while(sentIt.hasNext()) {
      int start = 0;
      double[] tempGrads = new double[this.thetaSize];
      DoubleMatrix word = this.dict.getRepresentation(sentIt.next());
      Iterator<Layer> layerIt = this.layers.iterator();
      DoubleMatrix rep = word;
      Stack<DoubleMatrix> input = new Stack<DoubleMatrix>();
      input.push(word);

      // fprop and store the representations at each layer in stack
      while(layerIt.hasNext()) {
        Layer l = layerIt.next();
        rep = l.fProp(rep);
        input.push(rep);
      }
      ListIterator<Layer> layerRevIt = this.layers.listIterator(this.layers.size());
      DoubleMatrix tempError = error;
      rep = input.pop();
      // backprop it
      while(layerRevIt.hasPrevious()) {
        Layer l = layerRevIt.previous();
        DoubleMatrix lGrads = l.bProp(rep, tempError);
        double[] lParamGrads = l.getParamGradients(input.peek(), lGrads);
        tempError = lGrads.mmul(l.getWeights().transpose());
        rep = input.pop();
        System.arraycopy(lParamGrads, 0, tempGrads, start, lParamGrads.length);
        start = l.getThetaSize();
      }
      grads.addi(new DoubleMatrix(1, this.thetaSize, tempGrads));
    }
    return grads;
  }
}
