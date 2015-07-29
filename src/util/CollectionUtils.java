package util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;

public class CollectionUtils {
	public static <K> void aggregateMaps(Map<K,Integer> A, Map<K,Integer> B) {
		for(K k : B.keySet()) {
			if(A.containsKey(k))
				A.put(k, A.get(k)+B.get(k));
			else
				A.put(k, B.get(k));
		}
	}
	
	public static <V> void appendMaps(Map<Integer,V> A, Map<Integer,V> B) {
		int index = A.size();
		for(int i: B.keySet()) {
			A.put(index, B.get(i));
			index++;
		}
	}
	public static <K,V> void printMap(Map<K,V> A, File file) throws FileNotFoundException, UnsupportedEncodingException {
		file.getParentFile().mkdirs();
		PrintWriter p = new PrintWriter(file, "UTF-8");
		for(K k: A.keySet())
			p.println(k+"\t"+A.get(k));
		p.close();
	}

	public static <K1, K2, V> void printMapOfMap(Map<K1, Map<K2,V>> A, File file) throws FileNotFoundException, UnsupportedEncodingException {
		file.getParentFile().mkdirs();
		PrintWriter p = new PrintWriter(file, "UTF-8");
		for(K1 k1: A.keySet())
			for(K2 k2: A.get(k1).keySet())
				p.println(k1+"\t"+k2+"\t"+A.get(k1).get(k2));
		p.close();
	}

	public static void main(String[] args) {
		Map<Integer, Integer> A = new HashMap<Integer, Integer>();
		Map<Integer, Integer> B = new HashMap<Integer, Integer>();
		
		Map<Integer, Map<Integer, Integer>> m1 = new HashMap<Integer, Map<Integer, Integer>>();
		Map<Integer, Map<Integer, Integer>> m2 = new HashMap<Integer, Map<Integer, Integer>>();
		m1.put(0, A);
		m2.put(0, B);
		
		A.put(1, 1);
		A.put(2, 1);
		
		B.put(1, 1);
		B.put(2, 1);
		B.put(3, 1);
		
		aggregateMaps(A,B);
		
		appendMaps(m1, m2);
		
		for(int i: m1.keySet())
			for(int j: m1.get(i).keySet())
				System.out.println(i+":"+j+":"+m1.get(i).get(j));
		
//		for(int i: A.keySet())
//			System.out.print(i+":"+A.get(i)+ " ");
		
	}
}
