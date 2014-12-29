package optim;

import optim.BasicGradientCalc;
import models.Model;
import common.Sentence;
import common.Corpus;

public class GradientCheck {
//  BasicGradientCalc gradFunc;
  NoiseGradientCalc gradFunc;
  public GradientCheck() {
//    gradFunc = new BasicGradientCalc();
    gradFunc = new NoiseGradientCalc();
  }
  public void optimise(Model model, Corpus c1, Corpus c2, Corpus c3) {
    this.gradFunc.setModel(model);
    for(int i=0; i<c1.getSize(); i++) {
      Sentence s1 = c1.get(i);
      Sentence s2 = c2.get(i);
      Sentence s3 = c3.get(i);

      gradFunc.setData(s1, s2, s3);
      System.out.printf("Processing Sentence: %d\ns1= %s \ns2 = %s\ns3 = %s\n\n", i, s1.toString(), s2.toString(), s3.toString());

      for(int j=0; j< model.getThetaSize(); j++) {
        double epsilon = 0.0001;
        
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
