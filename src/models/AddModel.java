package models;

import nn.Layer;

import org.jblas.DoubleMatrix;
import java.util.Iterator;
import java.util.ListIterator;

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
        l.fProp(temp);
        temp = l.getActivities();
      }
      rep.addi(temp);
    }
    return rep;
  }

  public void bProp(Sentence s1, Sentence s2, boolean add) {
    // send s2 up -- use it as label
    // send s1 up 
    // calculate error (1-2)
    // backpropagate error
    DoubleMatrix label = this.fProp(s2);
    DoubleMatrix pred = this.fProp(s1);

    DoubleMatrix error = pred.sub(label);
    ListIterator<Layer> layerIt = this.layers.listIterator(this.layers.size());
    while(layerIt.hasPrevious()) {
      Layer l = layerIt.previous();
      l.bProp(error);
      l.accumulateGradients(add);
      // CHECK BIAS ERROR
      error = l.getGradients().mmul(l.getWeights().transpose());
    }
  }

}
