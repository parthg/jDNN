package data;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.Iterator;

import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.ClassNotFoundException;
import java.io.ObjectInputStream;
import java.io.PrintWriter;

import data.Sample;
import data.Channel;
import data.AdHoc;
import random.RandomUtils;

import es.upv.nlel.corpus.NEWSDocType;
import es.upv.nlel.utils.CollectionUtils;
import es.upv.nlel.utils.Language;

public class SampleData {
  Map<Integer, String> data;
  Map<Integer, Sample> samples;
  Channel ch;

  public void loadData(String inFile) throws IOException {
    this.data = new HashMap<Integer, String>();
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), "UTF-8"));
    String line;
    int id = 0;
    while((line = br.readLine())!=null) {
      data.put(id, line.trim());
      id++;
    }
    br.close();
    System.out.println("Total " + this.data.size() + " data points loaded.");
  }

  public void loadSamples(String sampleObjFile) throws IOException, ClassNotFoundException {
    System.out.printf("Loading the samples from object file..\n");
    FileInputStream fis = new FileInputStream(sampleObjFile);
    ObjectInputStream ois = new ObjectInputStream(fis);

    this.samples = (Map<Integer, Sample>) ois.readObject();

    ois.close();
    fis.close();
  }

  public void getSampleStats() {
    int totPos = 0, totNeg = 0;
    int totDataPoints = this.data.size();
    for(int i: this.samples.keySet()) {
      totPos += this.samples.get(i).getPosSize();
      totNeg += this.samples.get(i).getNegSize();
    }
    System.out.printf("Total DataPoints = %d, Avg. Positive Samples = %.4f, Avg. Negative Samples = %.4f\n",totDataPoints, (double)((double)totPos/totDataPoints),(double) ((double)totNeg/totDataPoints));
  }

  public void prepareMatrixFiles(String dir, int n) throws IOException {
    dir += (dir.endsWith(File.separator)?"":File.separator);
    dir = dir.trim();
    PrintWriter p = new PrintWriter(dir+"bp-matrix.dat");
    PrintWriter pPos = new PrintWriter(dir+"bp-pos-matrix.dat");
    PrintWriter pNeg = new PrintWriter(dir+"bp-neg-matrix.dat");

    Map<Integer, String> newIdMap = new HashMap<Integer, String>();
    int id = 0;
    for(int i=0; i<this.samples.size(); i++) {

      Map<Integer, Integer> dp = ch.getVector(this.data.get(i));
      for(int j: dp.keySet())
        p.println(id+"\t"+j+"\t"+dp.get(j));
      
      Sample s = this.samples.get(i);
      Set<String> sp = s.getPos();
      Set<String> sn = s.getNeg();
      List<Map<Integer, Integer>> pList = new ArrayList<Map<Integer, Integer>>();
      List<Map<Integer, Integer>> nList = new ArrayList<Map<Integer, Integer>>();
      
      if(sp.size()>0 && sn.size()>0) {


        Iterator<String> it = sp.iterator();
        while(it.hasNext())
          pList.add(ch.getVector(it.next()));
      
        it = sn.iterator();
        while(it.hasNext())
          nList.add(ch.getVector(it.next()));


        if(pList.size()<n) {
          System.out.printf("[info] docid %d has %d positive samples hence inflating..\n",i, pList.size());
          randInflate(pList, n);
        }
        if(nList.size()<n) {
          System.out.printf("[info] docid %d has %d negative samples hence inflating..\n",i, nList.size());
          randInflate(nList, n);
        }

        // write off the lists
        for(int j=0; j<n; j++) {
          Map<Integer, Integer> inner = pList.get(j);
          for(int k: inner.keySet())
            pPos.println(id+"\t"+j+"\t"+k+"\t"+inner.get(k));
        }

        for(int j=0; j<n; j++) {
          Map<Integer, Integer> inner = nList.get(j);
          for(int k: inner.keySet())
            pNeg.println(id+"\t"+j+"\t"+k+"\t"+inner.get(k));
        }
        newIdMap.put(id, this.data.get(i));
        id++;
      }
    }

    p.close();
    pPos.close();
    pNeg.close();

    CollectionUtils.printMap(newIdMap, new File(dir+"latest-index-data.txt")); 

  }

  public static <K> void randInflate(List<K> list, int n) {
    while(list.size()<n)
      list.add(list.get(RandomUtils.nextInt(list.size())));
  }

  public static void main(String[] args) throws IOException, ClassNotFoundException {
    SampleData sd = new SampleData();
    if(args.length!=3) {
      System.out.printf("\n\nUsage: java data.SampleData <path_to_dataFile> <path_to_sample_object> <path_to_term_index>\n\n\twhere, data file is per-line sentence formatted file\n\n");
      System.exit(0);
    }

    sd.ch = new AdHoc();

		String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		sd.ch.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);

    sd.ch.loadTokenIndex(args[2]);

    sd.loadData(args[0]);
    sd.loadSamples(args[1]);
    sd.getSampleStats();

    sd.prepareMatrixFiles(new File(args[0]).getParent(), 2);
  }
}
