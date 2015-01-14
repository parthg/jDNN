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

    a = DMath.createMatrix(2, 4, new double[]{0, 1, 2,3, 4, 5, 6, 7});
    b = DMath.createMatrix(2, 4, new double[]{0, 1, 2, 3, 4,5, 6, 7});

    a.mul(b).print();
  
  }

  public static void testDivRows() {
    DMatrix a = DMath.createMatrix(3, 2, new double[]{0, 1, 2, 3, 4, 5});
    DMatrix b = a.mul(a);
    DMatrix col = b.sumColumns();
    b.print();
    col.print();
    b.divRowsi(col);
    b.print();
  }

  public static void testPow() {
    DMatrix a = DMath.createMatrix(3, 2, new double[]{0, 1, 2, 3, 4, 5});
    a.pow(2).print();
    a.powi(3);
    a.print();
  }

  public static void testSub() {
    DMatrix a = DMath.createMatrix(2,3, new double[]{1, 2, 3, 4, 5, 6});
    DMatrix b = DMath.createMatrix(2,3, new double[]{0, 1, 2, 3, 4, 5});

    a.sub(b).print();
    
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


  public static void testGemv() {
    DMatrix a = DMath.createMatrix(1, 2, new double[]{1, 2});
    DMatrix b = DMath.createMatrix(2, 4, new double[]{1, 2, 3, 4, 5, 6, 7, 8});

    a.mmul(b).print();
  }

  public static void testMMul() {
/*    DMatrix a = DMath.createOnesMatrix(2, 8);
    DMatrix b = DMath.createZerosMatrix(8, 2);
    DMatrix c = DMath.createMatrix(2, 2);

    a.mmul(b).print();*/

    DMatrix a = DMath.createMatrix(3,2, new double[]{1, 2, 3, 4, 5, 6});
    DMatrix b = DMath.createMatrix(2,4, new double[]{1, 2, 3, 4, 5, 6, 7, 8});

    System.out.printf("\nA*B (gemm)\nActual Answer:\n");
    System.out.printf("[11.000000 14.000000 17.000000 20.000000]\n");
    System.out.printf("[23.000000 30.000000 37.000000 44.000000]\n");
    System.out.printf("[35.000000 46.000000 57.000000 68.000000]\n");
  
    System.out.printf("Generated Answer:\n");
    a.mmul(b).print();

    a = DMath.createMatrix(4, 2, new double[]{0.298043, 0.140106,
       0.592947, 0.329253,
       0.092686, 0.873293,
       0.399121, 0.199589});
    b = DMath.createMatrix(2, 2, new double[]{0.481825, 0.860166,
       0.395595, 0.067174});

    System.out.printf("\nA*B (gemm)\nActual Answer:\n");
    System.out.printf("[0.19903  0.26578]\n[0.41595  0.53215]\n[0.39013  0.13839]\n[0.27126  0.35672]\n");
    
    System.out.printf("Generated Answer:\n");
    a.mmul(b).print();

    System.out.printf("\nA'*B (gemm)\nActual Answer:\n");
    System.out.printf("[21.000000 26.000000 31.000000 36.000000]\n[27.000000 34.000000 41.000000 48.000000]\n[33.000000 42.000000 51.000000 60.000000]\n");
    System.out.printf("Generated Answer:\n");
    a = DMath.createMatrix(2, 3, new double[]{1, 2, 3, 4, 5, 6});
    b = DMath.createMatrix(2, 4, new double[]{1, 2, 3, 4, 5, 6, 7, 8});

    a.mmul(true, false, b).print();

    a = DMath.createMatrix(3, 1, new double[]{1,2,3});
    b = DMath.createMatrix(3, 2, new double[]{1,2, 3, 4, 5, 6});
    System.out.printf("\nA'*B (gemv - inplace)\nActual Answer:\n22.000000 28.000000\n");
    System.out.printf("Generated Answer:\n");
    a.mmuli(true, false, b).print();

    a = DMath.createMatrix(3, 2, new double[]{1, 2, 3, 4, 5, 6});
    b= DMath.createMatrix(4, 2, new double[]{1, 2, 3, 4, 5, 6, 7, 8});
    System.out.printf("\nA*B' (gemm)\nActual Answer:\n[5.000000 11.000000 17.000000 23.000000 ]\n");
    System.out.printf("[11.000000 25.000000 39.000000 53.000000 ]\n");
    System.out.printf("[17.000000 39.000000 61.000000 83.000000 ]\nGenerated Answer:\n");
    
    a.mmul(false, true, b).print();

    b = DMath.createMatrix(4, 3, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    System.out.printf("\nA'*B' (gemm)\n[22.000000 49.000000 76.000000 103.000000 ]\n[28.000000 64.000000 100.000000 136.000000 ]\nGenerated Answer:\n");

    a.mmul(true, true, b).print();

//    c.print();
  }
  
  public static void testSumRows() {
    DMatrix a = DMath.createOnesMatrix(5, 5);
    a.sumRows().print();

    DMatrix b = DMath.createMatrix(5, 3, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    b.sumRows(1, 3).print();
  }

  public static void testSumColumns() {
    DMatrix a = DMath.createOnesMatrix(5, 5);
    a.sumColumns().print();

    DMatrix b = DMath.createMatrix(5, 3, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    b.sumColumns().sumRows().print();
  }
 

  public static void testFillWithArray() {
    DMatrix a = DMath.createOnesMatrix(3, 5);
    DMatrix b = DMath.createRandnMatrix(1, 5);

    
    b.print();

    a.fillWithArray(b).print();
    
//    DMatrix m = a.fillWithArray(b);
//    m.print();
  }

  public static void testFillRow() {
    DMatrix a = DMath.createMatrix(5, 5);
    DMatrix b = DMath.createOnesMatrix(1, 5);

    for(int i=0; i<5; i++) {
      a.fillRow(i, b.mul(i));
    }
    a.sumRows().print();
  }

  public static void testFillMatrix() {
    DMatrix a = DMath.createMatrix(6, 3);
    DMatrix b = DMath.createOnesMatrix(2, 3).mul(2);
    DMatrix c = DMath.createOnesMatrix(1, 3).mul(1);

    int row = 0;

    a.fillMatrix(row, b);
    row+=b.rows();
    a.fillMatrix(row, c);
    row+=c.rows();
    a.fillMatrix(row, b);
    row+=b.rows();
    a.fillMatrix(row, c);

    a.print();
  }

  public static void main(String[] args) {
    System.out.println(System.getProperty("use_cuda"));
//    testAddi();
//    testSub();
//    testMMuli();
//    testMMul();
//    testGemv();
//    testFillWithArray();

//    testMuli();
//    testPow();
//    testSumRows();
//    testSumColumns();
    testDivRows();
//    testFillWithArray();
//    testFillRow();
//    testFillMatrix();   
  }
}
