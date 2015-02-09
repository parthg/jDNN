package math;

import java.io.File;
import java.io.PrintWriter;
import java.io.IOException;

import java.util.Map;
import java.util.HashMap;

/** To store only non-zero elements
 */
public class SparseMatrix {
  int rows;
  int columns;
  Map<Integer, Map<Integer, Double>> data;

  public SparseMatrix(int r, int c) {
    this.rows = r;
    this.columns = c;
    this.data = new HashMap<Integer, Map<Integer, Double>>();
  }

  public int rows() {
    return this.rows;
  }

  public int columns() {
    return this.columns;
  }

  public void put(int r, int c, double v) {
    assert (r<this.rows && c<this.columns):System.out.printf("Indexes (%d, %d) bigger than Matrix size (%dx%d)", r, c, this.rows, this.columns);
    if(!this.data.containsKey(r)) {
      Map<Integer, Double> inner = new HashMap<Integer, Double>();
      inner.put(c, v);
      data.put(r, inner);
    }
    else {
      Map<Integer, Double> inner = this.data.get(r);
      inner.put(c, v);
      data.put(r, inner);
    }
  }

  public void print(File f) throws IOException {
    PrintWriter p = new PrintWriter(f);
    for(int i=0; i<this.rows; i++) {
      if(this.data.containsKey(i)) {
        for(int j: this.data.get(i).keySet()) {
          p.printf("%d\t%d\t%f\n", i, j, this.data.get(i).get(j));
        }
      }
    }
    p.close();
  }
}
