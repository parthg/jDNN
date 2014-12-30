package optim;

import java.util.List;
import java.util.ArrayList;

import optim.BasicGradientCalc;
import models.Model;
import common.Sentence;
import common.Corpus;

public class GradientCheck {
//  BasicGradientCalc gradFunc;
//  NoiseGradientCalc gradFunc;
  GradientCalc gradFunc;
  public GradientCheck() {
//    gradFunc = new BasicGradientCalc();
    gradFunc = new NoiseGradientCalc();
  }
  public void optimise(Model model, List<Corpus> c) {
    this.gradFunc.setModel(model);
    for(int i=0; i<c.get(0).getSize(); i++) {
      List<Sentence> s = new ArrayList<Sentence>();
      for(int corp=0; corp<c.size(); corp++)
        s.add(c.get(corp).get(i));

      gradFunc.setData(s);
      System.out.printf("Processing Sentence: %d\ns1= %s \ns2 = %s\ns3 = %s\n\n", i, s.get(0).toString(), s.get(1).toString(), s.get(2).toString());

      for(int j=0; j< model.getThetaSize(); j++) {
        double epsilon = 1e-7;
        
        gradFunc.setParameter(j, gradFunc.getParameter(j)+epsilon);
        double err1 = gradFunc.getValue();

        gradFunc.setParameter(j, gradFunc.getParameter(j)-2*epsilon);
        double err2 = gradFunc.getValue();

        double trueGrad = ((err1-err2)/(2*epsilon));

        gradFunc.setParameter(j, gradFunc.getParameter(j)+epsilon);
        double[] grads = new double[model.getThetaSize()];
        gradFunc.getValueGradient(grads);

        System.out.printf("True Grad: %.10f Calc Grad: %.10f\n", trueGrad, grads[j]);
      }
      
    }
  }
}
