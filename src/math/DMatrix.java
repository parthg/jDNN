package math;

public abstract class DMatrix {
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
    return this.data;
  }

  public void put(int i, double v) {
    assert (i<this.length);
    this.data[i] = v;
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

  public abstract DMatrix transpose();

  public abstract DMatrix add(DMatrix other);
  public abstract DMatrix addi(DMatrix other);

  public abstract DMatrix mul(DMatrix other);
  public abstract DMatrix muli(DMatrix other);

  public abstract DMatrix mul(double v);
  public abstract DMatrix muli(double v);

  public abstract DMatrix mmul(DMatrix other);
  public abstract DMatrix mmuli(DMatrix other);
  public abstract DMatrix mmuli(DMatrix other, DMatrix result);
}
