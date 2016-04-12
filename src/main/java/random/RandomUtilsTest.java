package random;

public class RandomUtilsTest {
  public static void testSuffleArray() {
    int[] arr = new int[10];
    for(int i=0; i<10; i++)
      arr[i] = i;

    RandomUtils.suffleArray(arr);
    for(int i=0; i< 10; i++)
      System.out.printf("%d ", arr[i]);

  }

  public static void main(String[] args) {
    testSuffleArray();
  }
}
