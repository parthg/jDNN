package common;

import java.util.Comparator;
import java.util.Arrays;

public class RankList {
  public static int[] rankList(double[] scores, int size) {
    if(size==0)
      size = scores.length;
    int N = Math.min(scores.length, size);
    int[] rl = new int[N];
    ArrayIndexComparator comparator = new ArrayIndexComparator(scores);
    Integer[] indexes = comparator.createIndexArray();
    Arrays.sort(indexes, comparator);
    for(int i=0; i<N; i++) {
      rl[i] = indexes[i];
    }
    return rl;
  }

  public static int rankPosition(double[] scores, int index) {
    assert (index<scores.length);
    ArrayIndexComparator comparator = new ArrayIndexComparator(scores);
    Integer[] indexes = comparator.createIndexArray();
    Arrays.sort(indexes, comparator);
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
