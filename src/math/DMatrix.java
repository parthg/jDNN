package math;

import jcuda.jcublas.JCublas;
import jcuda.Pointer;
import java.io.Closeable;
import math.jcublas.SimpleCuBlas;

public abstract class DMatrix implements Closeable {
//  int devId;
  boolean persist = false;
  Pointer cPointer = null; 
  int rows;
  int columns;
  int length;
  double[] data;

  protected int offset = 0;
  public DMatrix(int _rows, int _columns) {
    this.rows = _rows;
    this.columns = _columns;
    this.length = this.rows*this.columns;
    this.data = new double[this.length];
  }

  public DMatrix(int _rows, int _columns, double[] _data) {
    this.rows = _rows;
    this.columns = _columns;
    this.length = this.rows*this.columns;
    this.data = _data;
  }

  public double[] toArray() {
    double[] array = new double[this.length];
    System.arraycopy(this.data, 0, array, 0, this.length);
    return array;
  }

  public int index(int i, int j) {
    return i*rows+j;
  }
  public void put(int i, int j, double v) {
    this.data[index(i, j)] = v;
  }
  
  public void put(int i, double v) {
    assert (i<this.length);
    this.data[i] = v;
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

  public void resize(int newRows, int newColumns) {
    rows = newRows;
    columns = newColumns;
    length = newRows * newColumns;
    data = new double[rows * columns];
  }

  public void close() {
//    System.err.printf("close() in DMatrix\n");
    if(this.cPointer != null) {
      this.persist = false;
      JCublas.cublasFree(this.cPointer);
      this.cPointer = null;
    }
  }

  protected void finalize() {
//    System.err.printf("finalize() in DMatrix()\n");
    if(this.cPointer != null) {
      this.close();
    }
  }

  public void copyHtoD() {
    if(this.persist == false)
      this.persist = true;
    if(this.cPointer != null) {
      JCublas.cublasFree(this.cPointer);
      this.cPointer = null;
    }
  
    this.cPointer = SimpleCuBlas.alloc(this.data());
  }

  public void copyDtoH() {
    assert (this.cPointer!=null);
    SimpleCuBlas.getData(this,this.cPointer,Pointer.to(this.data()));
  }
  
  public void updateDeviceData() {
  }

  public void updateDeviceData(double[] newData) {
  }

  // currently its here, later override it with blas and cuda versions
  public DMatrix fillRow(int r, DMatrix arr) {
    assert (r<this.rows);
    System.arraycopy(arr.data(), 0, this.data, r*this.columns, arr.length());
    return this;
  }

  public DMatrix sumRows() {
    DMatrix sum = DMath.createZerosMatrix(1, this.columns());
    for(int i=0; i<this.length; i++) {
      sum.data()[i%this.columns] += (double) this.data[i];
    }
    return sum;
  }

  public abstract DMatrix transpose();

  public abstract DMatrix add(DMatrix other);
  public abstract DMatrix addi(DMatrix other);

  public abstract DMatrix add(double v);
  public abstract DMatrix addi(double v);

  public abstract DMatrix addi(double alpha, DMatrix other);

//  public abstract DMatrix addMuli(DMatrix A, DMatrix x);

  public abstract DMatrix sub(DMatrix other);
  public abstract DMatrix subi(DMatrix other);
  
  public abstract DMatrix mul(DMatrix other);
  public abstract DMatrix muli(DMatrix other);

  public abstract DMatrix mul(double v);
  public abstract DMatrix muli(double v);

  public abstract DMatrix mmul(boolean tA, boolean tB, DMatrix B);
  public abstract DMatrix mmul(DMatrix B);
  
  public abstract DMatrix mmuli(boolean tA, boolean tB, DMatrix B);
  public abstract DMatrix mmuli(DMatrix B);
  
  public abstract DMatrix mmul(boolean tA, boolean tB, DMatrix B, DMatrix C);
  public abstract DMatrix mmul(DMatrix B, DMatrix C); 

  public abstract DMatrix fillWithArray(DMatrix other);
}
