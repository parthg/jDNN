package random;
import java.util.Random;
import java.util.Set;
import java.util.HashSet;
import java.util.Iterator;

public class RandomUtils {
  private static Random r = new Random();
  public static void seed(long newSeed) {
    r = new Random(newSeed);
  }
  public static int nextInt(int max) {
    return r.nextInt(max);
  }
  public static int[] getRandomSamples(int max, int size) {
    Set<Integer> sample = new HashSet<Integer>();
    int count = 0;
    while(count<size) {
      int point = r.nextInt(max);
      if(!sample.contains(point)) {
        sample.add(point);
        count++;
      }
    }
    return setToArray(sample);
  }
  public static int[] setToArray(Set<Integer> integers) {
    int[] ret = new int[integers.size()];
    Iterator<Integer> iterator = integers.iterator();
    for (int i=0; i < ret.length; i++) {
      ret[i] = iterator.next().intValue();
    }
    return ret;
  }
}
