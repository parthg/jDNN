package models;

import math.DMatrix;
import math.DMath;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import es.upv.nlel.math.Matrix;

public class CanonicalCorrelation {
	DMatrix xCoef;
	DMatrix yCoef;
	
	DMatrix yCenter;
	
	public void loadXCoeff(String xPath) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(xPath));
		String line= "";
		String name = xPath.substring(xPath.lastIndexOf("/")+1);
		String[] cols = name.split("-");

		int rowDim = Integer.parseInt(cols[1].trim());
		int colDim = Integer.parseInt(cols[2].trim());
		this.xCoef = DMath.createMatrix(rowDim, colDim);
		
		int i=0;
		while((line = br.readLine())!=null) {
			cols = line.trim().split(" ");
			for(int j=0; j<colDim; j++) {
				this.xCoef.put(i, j, Double.parseDouble(cols[j].trim()));
			}
			i++;
		}
	}
	
	public void loadYCoeff(String yPath) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(yPath));
		String line= "";
		String name = new File(yPath).getName();
		String[] cols = name.split("-");
		int rowDim = Integer.parseInt(cols[1].trim());
		int colDim = Integer.parseInt(cols[2].trim());
		this.yCoef = DMath.createMatrix(rowDim, colDim);
		
		int i=0;
		while((line = br.readLine())!=null) {
			cols = line.trim().split(" ");
			for(int j=0; j<colDim; j++) {
				this.yCoef.put(i, j, Double.parseDouble(cols[j].trim()));
			}
			i++;
		}
	}
	
	public void loadYCenter(String yCenterPath) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(yCenterPath));
		String line= "";
		String name = new File(yCenterPath).getName();
		String[] cols = name.split("-");
		int colDim = Integer.parseInt(cols[2].trim());
		this.yCenter = DMath.createMatrix(1, colDim);
	
		while((line = br.readLine())!=null) {
			cols = line.trim().split(" ");
			for(int j=0; j<colDim; j++) {
				this.yCenter.put(j, Double.parseDouble(cols[j].trim()));
			}
		}
	}

  public DMatrix getXRep(DMatrix x) {
    return x.mmul(false, false, this.xCoef);
  }

  public DMatrix getYRep(DMatrix y) {
    return y.mmul(false, false, this.yCoef);
  }
	
	public DMatrix translateCanonical(DMatrix x) {
    DMatrix y = (x.mmul(false, false, this.xCoef)).mmul(false, true, this.yCoef);
    DMatrix yc = DMath.createMatrix(x.rows(), yCenter.columns());
    yc.fillWithArray(this.yCenter);
    y.addi(yc);
    return y;
/*		double[][] in = new double[1][];
		in[0] = x;
		double max = 0.0;
		double[] y = new double[this.yCenter.length];
		double[][] result = Matrix.multiplyTranspose(Matrix.multiply(in, this.xCoef), this.yCoef);
		for(int i=0; i< this.yCenter.length; i++) {
			y[i] = result[0][i]+this.yCenter[i];
			if(y[i]>max)
				max = y[i];
		}
		for(int i=0; i<y.length; i++) {
			y[i] = y[i]/max;
		}
		return y;*/
	}
	
	public static void main(String[] args) throws IOException {
		CanonicalCorrelation cca = new CanonicalCorrelation();
		cca.loadXCoeff("toy/enCoef-40-40");
		cca.loadYCoeff("toy/esCoef-40-40");
		cca.loadYCenter("toy/esCenter-1-40");
	}
}
