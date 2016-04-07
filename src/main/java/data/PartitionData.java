package data;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import es.upv.nlel.utils.Language;

public class PartitionData {
	Map<Integer, Map<Integer, Integer>> mat = new HashMap<Integer, Map<Integer, Integer>>();
  
	String outDir;
	long totSubFiles;
	long subSampleSize;
	
	public static void main(String[] args) throws IOException {
		PartitionData pd = new PartitionData();
		pd.loadData("data/fire/hi/data-matrix-sorted.dat");
		pd.outDir = "data/fire/hi/partition/";
		
		pd.calculateStats(10, pd.mat.size(), 55861);
		
		int[] randArr = pd.randPermute(pd.mat.size());
		pd.partitionData(randArr);
		
	}
	public void loadData(String inFile) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), "UTF-8"));
		String line = "";
		while((line = br.readLine())!=null) {
			String[] cols = line.split("\t");
			int i = Integer.parseInt(cols[0].trim());
			int j = Integer.parseInt(cols[1].trim());
			int count = Integer.parseInt(cols[2].trim());
		
      if(!mat.containsKey(i)) {
        Map<Integer, Integer> inner = new HashMap<Integer, Integer>();
        inner.put(j, count);
        mat.put(i, inner);
      }
      else {
        Map<Integer, Integer> inner = mat.get(i);
        inner.put(j, count);
        mat.put(i, inner);
      }	
		}
		br.close();
	}
	public void calculateStats(int maxGB, int total, int dim) {
		this.subSampleSize = getSubSampleSize(maxGB, dim);
		this.totSubFiles = (total/this.subSampleSize) + 1;
	}
  public static long getSubSampleSize(int maxGB, int dim) {
		double oneSampleSize = dim*8;
		long b = 1000000000;
		long a = maxGB * b;
		
		long tempSubSampleSize = (long) (a / oneSampleSize);
		long subSampleSize = tempSubSampleSize - (tempSubSampleSize%100);
    return subSampleSize;
  }
	/** It prints the indexes starting from 1.
	 * 
	 * @param outDir
	 * @param randArr
	 * @throws IOException
	 */
	public void partitionData(int[] randArr) throws IOException {
		FileOutputStream fos = null;
		PrintStream p = null;
		

		int file = 0;
		for(int i=1; i<=randArr.length; i++) {
			if(i%subSampleSize==1) {
				file++;
				if(file != 1) {
					p.close();
					fos.close();
					
				}
				fos = new FileOutputStream(this.outDir+"train-en-"+file+".txt");
				p = new PrintStream(fos);
				
			}
			for(int j: mat.get(randArr[i-1]).keySet())
				p.println(i+ "\t" + (j+1) + "\t" + mat.get(randArr[i-1]).get(j));
		}
		p.close();
		fos.close();
	}
	public int[] randPermute(int total) throws IOException {
		int[] a = new int[total];
		for(int i=0; i<total; i++)
			a[i] = i;
		this.suffleArray(a);
		FileOutputStream fos_map = new FileOutputStream(this.outDir+ "randMap.txt");
		PrintStream p_map = new PrintStream(fos_map, false, "UTF-8");
		for(int i=0; i<total; i++)
			p_map.println(i+"\t"+a[i]);
		p_map.close();
		fos_map.close();
		return a;
	}
	
	public void suffleArray(int[] arr) {
		Random rand = new Random();
		for(int i=arr.length-1; i>0; i--) {
			int index = rand.nextInt(i+1);
			int temp = arr[index];
			arr[index] = arr[i];
			arr[i] = temp;
		}
	}
}
