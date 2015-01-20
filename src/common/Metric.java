package common;
import java.util.Comparator;
import java.util.Arrays;
import math.DMatrix;
public class Metric {
  public static double mrr(DMatrix simMatrix) {
    assert (simMatrix.rows()==simMatrix.columns()):System.out.printf("Similarity matrix should be square matrix. Currently is : %d x %d", simMatrix.rows(), simMatrix.columns());
    double avgMRR = 0.0;
    for(int i=0; i<simMatrix.rows(); i++) {
      int pos = rankPosition(simMatrix.getRow(i).data(), i);
//      System.out.printf("pos = %d\n", pos);
      if(pos!=-1)
        avgMRR+= (1.0/(1+pos));
    }
    return (avgMRR/(double)simMatrix.rows());
  }

  public static int rankPosition(double[] scores, int index) {
    assert (index<scores.length);
    ArrayIndexComparator comparator = new ArrayIndexComparator(scores);
    Integer[] indexes = comparator.createIndexArray();
    Arrays.sort(indexes, comparator);
/*    for(int i=0; i<indexes.length; i++)
      System.out.printf("%d ", indexes[i]);
    System.out.printf("\n");*/
    for(int i=0; i<indexes.length; i++) {
      if(indexes[i] == index)
        return i;
    }
    return -1;
  }
}


// Class with comparator in decreasing order
class ArrayIndexComparator implements Comparator<Integer> {
  private final double[] array;
  ArrayIndexComparator(double[] _array) {
    this.array = _array;
  }
  public Integer[] createIndexArray() {
    Integer[] indexes = new Integer[this.array.length];
    for(int i = 0; i< indexes.length; i++) {
      indexes[i] = i;
    }
    return indexes;
  }
  @Override
  public int compare(Integer index1, Integer index2) {
    if(array[index1] < array[index2])
      return 1;
    else if(array[index1] > array[index2])
      return -1;
    else
      return 0;
//    return array[index1].compareTo(array[index2]);
  }
}
