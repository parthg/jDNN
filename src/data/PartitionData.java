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
	Map<Integer, Map<Integer, Integer>> enMat = new HashMap<Integer, Map<Integer, Integer>>();
//	Map<Integer, Map<Integer, Integer>> hiMat = new HashMap<Integer, Map<Integer, Integer>>();
	String outDir;
	long totSubFiles;
	long subSampleSize;
	
	public static void main(String[] args) throws IOException {
		PartitionData pd = new PartitionData();
//		pd.calculateStats(10, 171062, 36300);
		pd.loadData("data/fire/hi/data-matrix-sorted.dat", Language.EN);
		pd.outDir = "data/fire/hi/partition/";
//		pd.loadData("etc/hi_matrix.dat", Language.HI);
		
		pd.calculateStats(10, pd.enMat.size(), 55861);
		
		int[] randArr = pd.randPermute(pd.enMat.size());
		pd.partitionData(randArr);
		
	}
	public void loadData(String inFile, Language lang) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), "UTF-8"));
		String line = "";
		while((line = br.readLine())!=null) {
			String[] cols = line.split("\t");
			int i = Integer.parseInt(cols[0].trim());
			int j = Integer.parseInt(cols[1].trim());
			int count = Integer.parseInt(cols[2].trim());
			if(lang == Language.EN) {
				if(!enMat.containsKey(i)) {
					Map<Integer, Integer> inner = new HashMap<Integer, Integer>();
					inner.put(j, count);
					enMat.put(i, inner);
				}
				else {
					Map<Integer, Integer> inner = enMat.get(i);
					inner.put(j, count);
					enMat.put(i, inner);
				}
			}
/*			else if(lang == Language.HI){
				if(!hiMat.containsKey(i)) {
					Map<Integer, Integer> inner = new HashMap<Integer, Integer>();
					inner.put(j, count);
					hiMat.put(i, inner);
				}
				else {
					Map<Integer, Integer> inner = hiMat.get(i);
					inner.put(j, count);
					hiMat.put(i, inner);
				}
			}*/
				
		}
		br.close();
	}
	public void calculateStats(int maxGB, int total, int dim) {
		double oneSampleSize = dim*8;
//		double totBytes = total*dim*8;
//		double totGB = totBytes/1000000000;
//		totSubFiles = (int)(totGB/maxGB)+1;
		long b = 1000000000;
		long a = maxGB * b;
		
		long tempSubSampleSize = (long) (a / oneSampleSize);
		subSampleSize = tempSubSampleSize - (tempSubSampleSize%100);
		totSubFiles = (total/subSampleSize) + 1;
	}
	/** It prints the indexes starting from 1.
	 * 
	 * @param outDir
	 * @param randArr
	 * @throws IOException
	 */
	public void partitionData(int[] randArr) throws IOException {
		FileOutputStream fos_en= null, fos_hi= null;
		PrintStream p_en= null, p_hi= null;
		

		int file = 0;
		for(int i=1; i<=randArr.length; i++) {
			if(i%subSampleSize==1) {
				file++;
				if(file != 1) {
					p_en.close();
					fos_en.close();
					
/*					p_hi.close();
					fos_hi.close();*/
				}
				fos_en = new FileOutputStream(this.outDir+"train-en-"+file+".txt");
				p_en = new PrintStream(fos_en);
				
/*				fos_hi = new FileOutputStream(outDir+"train-hi-"+file+".txt");
				p_hi = new PrintStream(fos_hi);*/
			}
			for(int j: enMat.get(randArr[i-1]).keySet())
				p_en.println(i+"\t"+(j+1)+"\t"+enMat.get(randArr[i-1]).get(j));
/*			for(int j: hiMat.get(randArr[i-1]).keySet())
				p_hi.println(i+"\t"+(j+1)+"\t"+hiMat.get(randArr[i-1]).get(j));*/
		}
		p_en.close();
		fos_en.close();
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