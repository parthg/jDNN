package math;

public class DMathTest {
  public static void testAdd() {
    DMatrix a = DMath.createOnesMatrix(2,10);
    DMatrix b = DMath.createOnesMatrix(2,10);

    a.addi(b);

    a.print();
  }

  public static void testCudaAdd() {
  }
  public static void main(String[] args) {
    System.out.println(System.getProperty("use_cuda"));
    testAdd();
  }
}
