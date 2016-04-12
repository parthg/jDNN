import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertArrayEquals;

import math.DMatrix;
import math.DMath;


public class DMatrixTest {

  static final double DELTA = 0.0000001;

  @Test
  public void testMuli() {
//    System.setProperty("use_cuda", "true");
//    assertTrue("Please set use_cuda system property.", Boolean.parseBoolean(System.getProperty("use_cuda")));
    DMatrix a = DMath.createOnesMatrix(2,3);
    DMatrix b = DMath.createOnesMatrix(2,3);

    a.muli(2.0);
    assertArrayEquals(new double[]{2.0, 2.0, 2.0, 2.0, 2.0, 2.0}, a.data(), DELTA);

    a = DMath.createMatrix(2, 4, new double[]{0, 1, 2,3, 4, 5, 6, 7});
    b = DMath.createMatrix(2, 4, new double[]{0, 1, 2, 3, 4,5, 6, 7});

    // A.*B
    assertArrayEquals(new double[]{0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0}, a.mul(b).data(), DELTA);
  }
/*
  @Test
  public void testMulRows() {
    DMatrix a = DMath.createMatrix(3, 2, new double[]{0, 1, 2, 3, 4, 5});
    DMatrix b = a.mul(a);
//    b.print();
    DMatrix col = b.sumColumns();
//    col.print();
    a.mulRowsi(col);
    // Expected Output
    // [0.000000 1.000000 ]
    // [26.000000 39.000000 ]
    // [164.000000 205.000000 ]
//    a.print();
  }
  
  public static void testDivRows() {
    DMatrix a = DMath.createMatrix(3, 2, new double[]{0, 1, 2, 3, 4, 5});
    DMatrix b = a.mul(a);
    DMatrix col = b.sumColumns();
    a.divRowsi(col);
    // Expected output
    // [0.000000 1.000000 ]
    // [0.153846 0.230769 ]
    // [0.097561 0.121951 ]
    a.print();

    DMatrix c = DMath.createMatrix(3, 2);
    a.divRows(c.sumColumns()).print();

  }

  public static void testVectorNorm() {
    DMatrix a = DMath.createMatrix(3, 2, new double[]{0, 1, 2, 3, 4, 5});
    DMatrix b = a.vectorNorm();
    a.print("Mat1");
    b.print("Vector Norm of Mat1");

    a = DMath.createRandnMatrix(300, 128);
    b = a.vectorNorm();
    a.getRow(0).print("Mat2");
    b.getRow(0).print("Vector Norm of Mat2");
  }

  public static void testDotRows() {
    DMatrix a = DMath.createMatrix(2, 3, new double[]{0, 1, 2, 3, 4, 5});
    DMatrix b = DMath.createMatrix(2, 3, new double[]{1, 2, 3, 4, 5, 6});

    DMatrix c = a.dotRows(b);
    c.print();
  }

  public static void testPow() {
    DMatrix a = DMath.createMatrix(3, 2, new double[]{0, 1, 2, 3, 4, 5});
    a.pow(2.0).print();
    a.powi(3.0);
    a.print();
  }

  public static void testInv() {
    DMatrix a = DMath.createMatrix(3, 2, new double[]{0, 1, 2, 3, 4, 5});
    a.inv().print();

    a = DMath.createMatrix(1, 3, new double[]{0.57738, 0.57383, 0.58082});
    a.inv().print();
  }

  public static void testRowNorms() {
    DMatrix a = DMath.createMatrix(2, 4, new double[]{0, 1, 2, 3, 4, 5, 6, 7});
    a.rowNorms().inv().print();
  }
  

  public static void testSqrt() {
    DMatrix a = DMath.createMatrix(3, 2, new double[]{0, 1, 2, 3, 4, 5});
    a.sqrt().print();
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

  @Test
  public void testSumColumns() {
    DMatrix a = DMath.createOnesMatrix(5, 5);
    a.sumColumns().print();

    DMatrix b = DMath.createMatrix(5, 3, new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    b.sumColumns().print();
  }
 

  public static void testFillWithArray() {
    DMatrix a = DMath.createOnesMatrix(3, 5);
    DMatrix b = DMath.createRandnMatrix(1, 5);

    
    b.print();

    a.fillWithArray(b).print();
    
//    DMatrix m = a.fillWithArray(b);
//    m.print();
  }

  public static void testGetRow() {
    DMatrix a = DMath.createMatrix(3, 2, new double[]{1, 2, 3, 4, 5, 6});
    a.getRow(0).print();
    
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

  public static void testConcatVertically() {
    DMatrix a = DMath.createMatrix(2, 5, new double[]{1,2,3,4,5,6,7,8,9,10});
    DMatrix b = DMath.createMatrix(1, 5, new double[]{11, 12, 13, 14, 15});

    a.print("A");
    b.print("B");
    a.concatVertically(b);
    a.print("A (concat) B");
  }

  public static void testTruncateRows() {
    DMatrix a = DMath.createMatrix(5, 3);
    DMatrix b = DMath.createMatrix(3, 3, new double[]{1,2,3,4,5,6,7,8,9});
    a.fillMatrix(0, b);
    a.print("A");
    a.truncateRows(3, 3);
    a.print("A (truncated)");
  }

  @Test
  public void testInflateRows() {
    DMatrix a = DMath.createMatrix(3, 3, new double[]{1,2,3,4,5,6,7,8,9});
    a.print("A");
    a.inflateRows(5, 3);
    a.print("A (Inflated)");
  }
*/

/*  public static void main(String[] args) {
    System.out.println(System.getProperty("use_cuda"));
//    testAddi();
//    testSub();
//    testMMuli();
//    testMMul();
//    testGemv();
//    testFillWithArray();

    testMuli();
//    testPow();
//    testSumRows();
//    testSumColumns();
//    testInv();
//    testGetRow();
//    testRowNorms();
//    testSqrt();
//    testDivRows();
//    testMulRows();
//    testDotRows();
//    testVectorNorm();
//    testFillWithArray();
//    testFillRow();
//    testFillMatrix();   
//    testConcatVertically();
//    testTruncateRows();
//    testInflateRows();
  }*/
}
