package models;

import math.DMath;
import math.DMatrix;

import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.IOException;

public class LinearModel {
  int inSize;
  int outSize;
  DMatrix operator;

  public LinearModel(int _inDim, int _outDim) {
    this.inSize = _inDim;
    this.outSize = _outDim;
  }

  public int inSize() {
    return this.inSize;
  }

  public int outSize() {
    return this.outSize;
  }

  public void load(File modelFile) throws IOException {
    this.operator = DMath.createMatrix(this.inSize, this.outSize);
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(modelFile)));
    String line = "";
    int row=0;
    while((line = br.readLine())!=null) {
      String[] cols = line.split(" ");
      for(int j=0; j<this.outSize; j++) {
        this.operator.put(row, j, Double.parseDouble(cols[j]));
      }
      row++;
    }
    br.close();
  }

  public DMatrix project(DMatrix input) {
    return input.mmul(false, false, this.operator);
  }
}
