package math;

public class DMathTest {


  public static void testMuli() {
    DMatrix a = DMath.createOnesMatrix(2,5);
    DMatrix b = DMath.createOnesMatrix(2,5);

    a.muli(2.0);
    a.print();
    b.muli(2.0);
    a.muli(b);
    a.print();
  }

  public static void testAddi() {
    DMatrix a = DMath.createOnesMatrix(2,5);
    DMatrix b = DMath.createOnesMatrix(2,5);

    a.addi(b);
    a.print();
  }

  public static void testMMuli() {
    DMatrix a = DMath.createOnesMatrix(2, 2);
    DMatrix b = DMath.createOnesMatrix(2, 2);

    a.mmuli(b);
    a.print();
  }

  public static void testMMul() {
    DMatrix a = DMath.createOnesMatrix(2, 80000);
    DMatrix b = DMath.createOnesMatrix(80000, 2);
    DMatrix c = DMath.createMatrix(2, 2);

    a.mmuli(b, c);
    c.print();
  }
  
  public static void main(String[] args) {
    System.out.println(System.getProperty("use_cuda"));
    testAddi();
    testMMuli();
    testMMul();

    testMuli();
  }
}
