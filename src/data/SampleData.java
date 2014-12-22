package data;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;
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

  List<Map<Integer, Integer>> dataMatrix;
  List<List<Map<Integer, Integer>>> posMatrix;
  List<List<Map<Integer, Integer>>> negMatrix; 
  Channel ch;

  static boolean inflate = true;
  static boolean randomise = true;

  public void loadData(String inFile) throws IOException {
    this.data = new HashMap<Integer, String>();
    this.dataMatrix = new ArrayList<Map<Integer, Integer>>();
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), "UTF-8"));
    String line;
    int id = 0;
    while((line = br.readLine())!=null) {
      data.put(id, line.trim());
      Map<Integer, Integer> dp = ch.getVector(line.trim());
      this.dataMatrix.add(dp);
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

    this.posMatrix = new ArrayList<List<Map<Integer, Integer>>>();
    this.negMatrix = new ArrayList<List<Map<Integer, Integer>>>();

    for(int i=0; i<this.samples.size(); i++) {
      Sample s = this.samples.get(i);
      
      // pos samples 
      List<Map<Integer, Integer>> sampleBuffer = new ArrayList<Map<Integer, Integer>>();
      Iterator<String> it = s.getPos().iterator();
      while(it.hasNext()) {
        sampleBuffer.add(ch.getVector(it.next()));
      }
      this.posMatrix.add(sampleBuffer);

      // neg samples
      sampleBuffer = new ArrayList<Map<Integer, Integer>>();
      it = s.getNeg().iterator();
      while(it.hasNext()) {
        sampleBuffer.add(ch.getVector(it.next()));
      }
      this.negMatrix.add(sampleBuffer);
    }
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

  public void prepareMatrixFiles(String dir, int n, int maxGB, int dim) throws IOException {
    long subSampleSize = PartitionData.getSubSampleSize(maxGB, dim);
    dir += (dir.endsWith(File.separator)?"":File.separator);
    dir = dir.trim()+"partition/";

    new File(dir).mkdirs();

    if(this.dataMatrix.size() != this.posMatrix.size() || this.dataMatrix.size() != this.negMatrix.size()) {
      System.out.printf("The data size does not match. dataSize = %d posSize = %d negSize = %d. Exiting.. \n", this.dataMatrix.size(), this.posMatrix.size(), this.negMatrix.size());
      System.exit(0);
    }
    PrintWriter p = null, pPos = null, pNeg = null, pData = null;
    
    int id= 0;
    // print data
    int[] indexes = new int[dataMatrix.size()];
    Map<Integer, Integer> randMap = new HashMap<Integer, Integer>();
    Set<Integer> indexTestSet = new HashSet<Integer>();
    for(int i =0; i<indexes.length; i++)
      indexTestSet.add(i);
    if(randomise) {
      randMap = RandomUtils.randArray(dataMatrix.size());
      for(int i=0; i< indexes.length; i++) {
        indexes[i] = randMap.get(i);
        indexTestSet.remove(indexes[i]);
      }
    }
    else {
      for(int i=0; i< indexes.length; i++)
        indexes[i]=i;
    }

    if(indexTestSet.size()==0)
      System.out.println("Indexes correctly randomised.");
    else
      System.out.println("Indexes not correctly randomised.");

    int file = 0;
    for(int i=0; i<this.dataMatrix.size(); i++) {
      if((id+1)%subSampleSize==1) {
        file++;
        // if its not the first file
        if(file != 1) {
          p.close(); 
          pPos.close();
          pNeg.close();
          pData.close();
        }

        p = new PrintWriter(dir+"train-bp-matrix-"+file+".txt");
        pPos = new PrintWriter(dir+"bp-pos-matrix-"+file+".txt");
        pNeg = new PrintWriter(dir+"bp-neg-matrix-"+file+".txt");
        pData = new PrintWriter(dir+"bp-text-data-"+file+".txt");
      }
      Map<Integer, Integer> dp = this.dataMatrix.get(indexes[i]);
      List<Map<Integer, Integer>> pList = this.posMatrix.get(indexes[i]);
      List<Map<Integer, Integer>> nList = this.negMatrix.get(indexes[i]);

      if(pList.size()>0 && nList.size()>0) {
        for(int k: dp.keySet())
          p.println(id+"\t"+k+"\t"+dp.get(k));
    
    
        // print pos        
        if(pList.size()<n && inflate) {
          System.out.printf("[info] docid %d has %d positive samples hence inflating..\n",indexes[i], pList.size());
          randInflate(pList, n);
        }
        if(nList.size()<n && inflate) {
          System.out.printf("[info] docid %d has %d negative samples hence inflating..\n",indexes[i], nList.size());
          randInflate(nList, n);
        }
        // print pos
        for(int j=0; j<n; j++) {
          Map<Integer, Integer> inner = pList.get(j);
          for(int k: inner.keySet()) {
//            pPos.println(id+"\t"+j+"\t"+k+"\t"+inner.get(k));
            pPos.println(((n*id)+j)+"\t"+k+"\t"+inner.get(k));
          }
        }

        // pring neg
        for(int j=0; j<n; j++) {
          Map<Integer, Integer> inner = nList.get(j);
          for(int k: inner.keySet()) {
//            pNeg.println(id+"\t"+j+"\t"+k+"\t"+inner.get(k));
            pNeg.println(((n*id)+j)+"\t"+k+"\t"+inner.get(k));
          }
        }
        pData.println(id+"\t"+this.data.get(indexes[i]));
        id++; 
      }
    }
    
    p.close();
    pPos.close();
    pNeg.close();
    pData.close();
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

    sd.prepareMatrixFiles(new File(args[0]).getParent(), 1, 10, 55861);
  }
}
