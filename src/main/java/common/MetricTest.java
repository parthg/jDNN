package common;

import math.DMath;
import math.DMatrix;

public class MetricTest {
  public static void testMRR() {
    DMatrix sim = DMath.createMatrix(3, 3, new double[]{0.1, 0.2, 0.3, 0.2, 0.1, 0.3, 0.3, 0.1, 0.2});

    // Ans = 0.388889 
    System.out.printf("MRR = %.6f\n", Metric.mrr(sim));
  }

  public static void testRankList() {
    DMatrix sim = DMath.createMatrix(3, 3, new double[]{0.1, 0.2, 0.3, 0.2, 0.1, 0.3, 0.3, 0.1, 0.2});
    int[] rl = RankList.rankList(sim.getRow(1).data(), 0);
    for(int i=0; i<rl.length; i++) {
      System.out.printf("%d ", rl[i]);
    }

  }
  public static void main(String[] args) {
//    testMRR();
    testRankList();
  }
}
