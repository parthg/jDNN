package optim;

import java.util.List;
import java.util.ArrayList;

import optim.BasicGradientCalc;
import models.Model;
import common.Sentence;
import common.Corpus;

public class GradientCheck {
  GradientCalc gradFunc;
  public GradientCheck(GradientCalc _gradFunc) {
    gradFunc = _gradFunc;
  }
  public void optimise(Model model) {
    this.gradFunc.setModel(model);

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
