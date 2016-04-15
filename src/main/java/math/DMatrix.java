package math;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.IOException;

import jcuda.jcublas.JCublas;
import jcuda.Pointer;
import java.io.Closeable;
import math.jcublas.SimpleCuBlas;

/** DMatrix is row-major order matrix.
 *
 * passing [1 2 3 4] to 2x2 matrix will form following matrix.
 * [
 * 1  2
 * 3  4
 * ]
 */
public abstract class DMatrix implements Closeable {
//  int devId;
  boolean persist = false; // flag to keep the matrix on GPU
  Pointer cPointer = null; 
  int rows;
  int columns;
  int length;
  double[] data;

  protected int offset = 0;
 
	/** Constructs the matrix with the elements initialised with zeros.
 */ 
  public DMatrix(int _rows, int _columns) {
    this.rows = _rows;
    this.columns = _columns;
    this.length = this.rows*this.columns;
    this.data = new double[this.length];
  }

	/** Constructs with the specific data array 
 */
  public DMatrix(int _rows, int _columns, double[] _data) {
    this.rows = _rows;
    this.columns = _columns;
    this.length = this.rows*this.columns;
    this.data = _data;
  }

	/** Copies the elements to a new array and returns a copy of it.
 */
  public double[] toArray() {
    double[] array = new double[this.length];
    System.arraycopy(this.data, 0, array, 0, this.length);
    return array;
  }

  public void printDim(String name) {
    System.out.printf("%S: %d x %d\n", name, this.rows, this.columns);
  }

  /** returns the array index for the given matrix position (row, column).
   */
  public int index(int i, int j) {
    return i*columns+j;
  }

  public void put(int i, int j, double v) {
    assert (i<this.rows && j<this.columns);
    this.data[index(i, j)] = v;
  }
  
  public void put(int i, double v) {
    assert (i<this.length);
    this.data[i] = v;
  }

  public double get(int i, int j) {
    assert (i<this.rows && j<this.columns);
    return this.data[index(i, j)];
  }

  public double get(int i) {
    assert (i<this.length);
    return this.data[i];
  }

  public Pointer pointer() {
    return this.cPointer;
  }

  public boolean persist() {
    return this.persist;
  }

  public int offset() {
    return this.offset;
  }

  public int rows() {
    return this.rows;
  }

  public int columns() {
    return this.columns;
  }

  public int length() {
    return this.length;
  }

  public double[] data() {
    return this.data;
  }

  public int elemSize() {
    return 8;
  }

  /** Sum of squared distance
   */
  public double squaredDistance(DMatrix other) {
    assert (this.length == other.length());
    double sd = 0.0;
    for (int i = 0; i < length; i++) {
      double d = get(i) - other.get(i);
      sd += d * d;
    }
    return sd;
  }

  public double distance2(DMatrix other) {
    return (double) Math.sqrt(squaredDistance(other));
  }

  public void print() {
    for(int i= 0; i<this.rows; i++) {
      System.out.printf("[");
      for(int j = 0; j<this.columns; j++) {
        System.out.printf("%f ", this.data[i*this.columns+j]);
      }
      System.out.printf("]\n");
    }
  }

  public void print(String comment) {
    System.out.printf("\n%s\n", comment);
    for(int i= 0; i<this.rows; i++) {
      System.out.printf("[");
      for(int j = 0; j<this.columns; j++) {
        System.out.printf("%f ", this.data[i*this.columns+j]);
      }
      System.out.printf("]\n");
    }
  }

  public void printToFile(File f) throws IOException {
    FileOutputStream fos = new FileOutputStream(f);
    PrintStream p = new PrintStream(fos);
    for(int i= 0; i<this.rows; i++) {
      for(int j = 0; j<this.columns; j++) {
        p.printf("%f ", this.data[i*this.columns+j]);
      }
      p.printf("\n");
    }
    p.close();
    fos.close();
  }
  
  /** Changes the matrix to the new dimensions. Careful: It overwrites the original matrix.
   */
  public void resize(int newRows, int newColumns) {
    // TODO: Do things for GPU related stuff.
    rows = newRows;
    columns = newColumns;
    length = newRows * newColumns;
    data = new double[rows * columns];
  }

  /** Frees the matrix from the GPU and clears the Pointer.
   */
  public void close() {
//    System.err.printf("close() in DMatrix\n");
    if(this.cPointer != null) {
      this.persist = false;
      SimpleCuBlas.free(this.cPointer);
//      JCublas.cublasInit();
//      JCublas.cublasFree(this.cPointer);
      this.cPointer = null;
    }
  }

  protected void finalize() {
//    System.err.printf("finalize() in DMatrix()\n");
    if(this.cPointer != null) {
      this.close();
    }
  }

  /** Makes a device copy of the matrix.
   */
  public void copyHtoD() {
    if(this.persist == false)
      this.persist = true;
    if(this.cPointer != null) {
      SimpleCuBlas.free(this.cPointer);
//      JCublas.cublasFree(this.cPointer);
      this.cPointer = null;
    }
    this.cPointer = SimpleCuBlas.alloc(this.data());
  }

  /** Copies the matrix from device to host.
   */
  public void copyDtoH() {
    assert (this.cPointer!=null);
    SimpleCuBlas.getData(this,this.cPointer,Pointer.to(this.data()));
  }
  
  // TODO
  public void updateDeviceData() {
  }

  // TODO
  public void updateDeviceData(double[] newData) {
  }

  // currently its here, later override it with blas and cuda versions
  public DMatrix fillRow(int r, DMatrix arr) {
    assert (r<this.rows);
    System.arraycopy(arr.data(), 0, this.data, r*this.columns, arr.length());
    return this;
  }

  public DMatrix fillRow(int r, int howMany, DMatrix arr) {
    assert (r+howMany<=this.rows);
    int start = r;
    for(int i=0; i<howMany; i++) {
      System.arraycopy(arr.data(), 0, this.data, start*this.columns, arr.length());
      start ++;
    }
    return this;
  }
  
  public DMatrix fillMatrix(int r, DMatrix mat) {
    assert (r<this.rows && mat.columns()==this.columns());
    System.arraycopy(mat.data(), 0, this.data, r*this.columns, mat.length());
    return this;
  }

  public DMatrix sumRows() {
    DMatrix sum = DMath.createZerosMatrix(1, this.columns());
    for(int i=0; i<this.length; i++) {
      sum.data()[i%this.columns] += (double) this.data[i];
    }
    return sum;
  }

  public DMatrix sumRows(int startRow, int howMany) {
    return null;
  }

  public DMatrix sumColumns() {
    return null;
  }

  public DMatrix getRow(int r) {
    double[] rowData = new double[this.columns];
    System.arraycopy(this.data(), r*this.columns(), rowData, 0, this.columns());
    return DMath.createMatrix(1, this.columns(), rowData);
  }

  public DMatrix concatVertically(DMatrix B) {
    assert (this.columns == B.columns()): System.out.printf("Matrices don't have same number of columns (" + this.columns + " != " + B.columns() + ".");

    double[] newData = new double[this.length+B.length()];

    System.arraycopy(this.data, 0, newData, 0, this.length);
    System.arraycopy(B.data(), 0, newData, this.length, B.length());

    this.rows = this.rows+B.rows();
    this.length = this.rows*this.columns;
    this.data = newData;

    return this;
  }

  public void truncateRows(int _rows, int _columns) {
    if(this.rows == _rows)
      return;
    assert (this.columns == _columns && this.rows > _rows);
    double[] newData = new double[_rows*_columns];
    System.arraycopy(this.data, 0, newData, 0, newData.length);

    this.rows = _rows;
    this.columns = _columns;
    this.length = this.rows*this.columns;
    this.data = newData;
  }

  public void inflateRows(int _rows, int _columns) {
    assert (this.columns == _columns && this.rows< _rows);
    double[] newData = new double[_rows*_columns];
    System.arraycopy(this.data, 0, newData, 0, this.length);

    this.rows = _rows;
    this.columns = _columns;
    this.length = this.rows*this.columns;
    this.data = newData;
  }

//  public abstract DMatrix transpose();

  public abstract DMatrix add(DMatrix other);
  public abstract DMatrix addi(DMatrix other);

  public abstract DMatrix add(double v);
  public abstract DMatrix addi(double v);

  public abstract DMatrix addi(double alpha, DMatrix other);

//  public abstract DMatrix addMuli(DMatrix A, DMatrix x);

  public abstract DMatrix inv();
  public abstract DMatrix invi();
  
  public abstract DMatrix sub(DMatrix other);
  public abstract DMatrix subi(DMatrix other);
  
  public abstract DMatrix sub(double v);
  public abstract DMatrix subi(double v);
  
  public abstract DMatrix sqrt();
  public abstract DMatrix sqrti();
  
  public abstract DMatrix mul(DMatrix other);
  public abstract DMatrix muli(DMatrix other);

  public abstract DMatrix mul(double v);
  public abstract DMatrix muli(double v);
  

  public abstract DMatrix pow(double v);
  public abstract DMatrix powi(double v);

  public abstract DMatrix mmul(boolean tA, boolean tB, DMatrix B);
  public abstract DMatrix mmul(DMatrix B);
  
  public abstract DMatrix mmuli(boolean tA, boolean tB, DMatrix B);
  public abstract DMatrix mmuli(DMatrix B);
  
  public abstract DMatrix mmul(boolean tA, boolean tB, DMatrix B, DMatrix C);
  public abstract DMatrix mmul(DMatrix B, DMatrix C); 

  public abstract DMatrix fillWithArray(DMatrix other);

  public abstract DMatrix mulRows(DMatrix colVector);
  public abstract DMatrix mulRowsi(DMatrix colVector);
  
  public abstract DMatrix divRows(DMatrix colVector);
  public abstract DMatrix divRowsi(DMatrix colVector);

  /** Return the the matrix where each row is unit based on the L2 norm.
   */
  public abstract DMatrix vectorNorm();

  /** Calculates the L2 norm of each row and returns as the column vector.
   */
  public abstract DMatrix rowNorms();
  
  
  /** Calcuates the dot product between the rows of this and B and returns as column Vector
   */
  public abstract DMatrix dotRows(DMatrix B);
}
