package random;
import java.util.Random;
import java.util.Set;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.HashMap;

import java.io.IOException;

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
  
	public static Map<Integer, Integer> randArray(int total) throws IOException {
		int[] a = new int[total];
		for(int i=0; i<total; i++)
			a[i] = i;
		suffleArray(a);
    Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for(int i=0; i<total; i++)
        map.put(i, a[i]);
		return map;
	}
	
  public static void suffleArray(int[] arr) {
		Random rand = new Random();
		for(int i=arr.length-1; i>0; i--) {
			int index = rand.nextInt(i+1);
			int temp = arr[index];
			arr[index] = arr[i];
			arr[i] = temp;
		}
	}
}
