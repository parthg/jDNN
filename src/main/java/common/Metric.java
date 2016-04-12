package common;

import math.DMatrix;

public class Metric {
  public static double mrr(DMatrix simMatrix) {
    assert (simMatrix.rows()==simMatrix.columns()):System.out.printf("Similarity matrix should be square matrix. Currently is : %d x %d", simMatrix.rows(), simMatrix.columns());
    double avgMRR = 0.0;
    for(int i=0; i<simMatrix.rows(); i++) {
      int pos = RankList.rankPosition(simMatrix.getRow(i).data(), i);
      if(pos!=-1)
        avgMRR+= (1.0/(1+pos));
    }
    return (avgMRR/(double)simMatrix.rows());
  }
  
  public static double map(DMatrix simMatrix) {
    assert (simMatrix.rows()==simMatrix.columns()):System.out.printf("Similarity matrix should be square matrix. Currently is : %d x %d", simMatrix.rows(), simMatrix.columns());
    double avgAP = 0.0;

    return avgAP;
  }
}
