package models;

import common.Dictionary;
import nn.Layer;
import nn.LogisticLayer;
import math.DMath;
import math.DMatrix;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.Stack;

import common.Sentence;

public class Autoencoder extends Model {
  public Autoencoder() {
    super();
  }
  
  public void load(String modelFile, Dictionary _dict) throws IOException {
    System.out.printf("Loading the Autoencoder.\n");
    this.setDict(_dict);
    
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(modelFile)));
    String line = "";
    while((line = br.readLine())!=null) {
      if(line.startsWith("#numLayers")) {
        String[] cols = line.split("=");
        int nLayers = Integer.parseInt(cols[1].trim());
        for(int lId = 0; lId<nLayers; lId++) {
          String lDetails = br.readLine();
          String[] lCols = lDetails.split("=");
          String[] lSize = lCols[1].trim().split(" ");
          int lLength = Integer.parseInt(lSize[1].trim());
          Layer l = new LogisticLayer(lLength);
          this.addHiddenLayer(l);
        }
      }
      else if(line.startsWith("#params")) {
        this.printArchitecture();
        this.init();
        double[] params = new double[this.thetaSize];
        String[] cols = line.split("=");
        Scanner sc = new Scanner(cols[1].trim());
        int i=0;
        while(sc.hasNextDouble()) {
          params[i] = sc.nextDouble();
          i++;
        }
        this.setParameters(params);
      }
    }
    br.close();
  }
  
  public DMatrix fProp(Sentence sent) {
    return null;
  } 

  public DMatrix getRepresentation(DMatrix sentMatrix) {
    return null;
  }

  public DMatrix fProp(DMatrix input) {
    DMatrix temp = this.layers.get(0).fProp(input);
    if(this.getNumLayers()>1) {
      for(int i=1; i<getNumLayers(); i++) {
        temp = this.layers.get(i).fProp(temp);
      }
    }
    return temp;
  }
  
  public DMatrix bProp(Sentence s1, Sentence s2) {
    return null;
  }
  public DMatrix bProp(Sentence s1, DMatrix error) {
    return null;
  }
  
  /* input  = matrix
   * rep    = matrix
   * error  = matrix
   */
  public DMatrix bProp(DMatrix input, DMatrix rep, DMatrix error) { 
    return null;
  }
  
  public DMatrix bProp(DMatrix input, DMatrix error) {
    return null;
  }
}
