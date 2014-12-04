package retrieval;

import java.io.File;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

import java.io.IOException;

import java.util.Map;
import java.util.HashMap;
import java.lang.Math;

import es.upv.nlel.wrapper.TerrierWrapper;
import org.terrier.matching.ResultSet;
import es.upv.nlel.utils.FileIO;
import data.Sample;
import random.RandomUtils;

import antlr.RecognitionException;
import antlr.TokenStreamException;


public class RetrieveSamples {
  TerrierWrapper terrier;
  public boolean verbose = false;
  public static void main(String[] args) throws IOException, RecognitionException, TokenStreamException {

    RetrieveSamples ret = new RetrieveSamples();

    Map<Integer, Sample> sampleMap = new HashMap<Integer, Sample>();

    String terrierPath = "/home/parth/workspace/terrier-3.5/";
    String lang = "hi";
    boolean sw_removal = true;
    boolean stem = true;

    ret.terrier = new TerrierWrapper(terrierPath);
    String dataFile = "etc/data/fire/hi/title-only.txt";
    String tempSentDir = "data/fire/hi/sent/";

    if(!new File(tempSentDir).exists())
      (new File(tempSentDir)).mkdirs();


    String indexPath = terrierPath+"var/index/firesent/";
    ret.terrier.setIndex(indexPath, lang);
    ret.terrier.setLanguage(lang);

    if(!new File(indexPath+lang+".docid.map").exists()) {
      System.out.println("Preparing Index...");
      int totFiles = (new File(tempSentDir)).list().length;
      if(totFiles>0) {
        System.out.println("There can not be any file in the tempSentDir, rather there exist " + totFiles +" files. Please delete them.. Exiting..");
        System.exit(0);
      }
      ret.terrier.prepareSentenceIndex(dataFile, ".txt", lang, sw_removal, stem, tempSentDir);
    }
    else {
      if(sw_removal)
        ret.terrier.setStopwordRemoval(lang);
      if(stem)
        ret.terrier.setStemmer(lang);
      ret.terrier.learnDocName();
      ret.terrier.loadIndex(indexPath, lang, lang);
      
    }

    System.out.println("Drawing Samples...");
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(dataFile), "UTF-8"));
    String line = "";
    
//    Map<Integer, String> dataIds = new HashMap<Integer, String>();

    int sampleSize = 4;
    int id = 0;
    while((line = br.readLine())!=null) {
      Sample sample = ret.getSample(line.trim(), sampleSize, sampleSize, tempSentDir);
//      if(sample.getPosSize==sampleSize && sample.getNegSize == sampleSize) {
        sampleMap.put(id, sample);
//        dataIds.put(id, line.trim());
        id++;
//      }
    }
    br.close();
    // read query file line by line
    // retrieve relevant and non-relevant document and encapsulate them in Sample
    // write in a proper form
    System.out.println("Writing the Object");
    FileOutputStream fos = new FileOutputStream("obj/hi-samples.obj");
    ObjectOutputStream oos = new ObjectOutputStream(fos);

    oos.writeObject(sampleMap);

    oos.close();
    fos.close();


  }
  public Sample getSample(String txt, int pos, int neg, String dataDir) throws IOException, RecognitionException, TokenStreamException {
    txt = this.terrier.pipelineText(txt);
    ResultSet rs = this.terrier.getResultSet(txt, "TF_IDF", false, 0);
    int[] docid = rs.getDocids();
    double[] scores = rs.getScores();

    Sample sample = new Sample();

    if(this.verbose)
      System.out.println("Input Text = " + txt);

    int p = 0, i=0, n=0;
    pos = Math.min(pos,docid.length-1);
    while(sample.getPosSize()<pos && i<docid.length-1) {
      i++;
      String sentId = this.terrier.getDocName(docid[i]);
      String s = FileIO.fileToString(new File(dataDir+sentId));
      if(!txt.equals(s)) {
        if(this.verbose)
          System.out.println("\t\tPos " + (i+1) + " " + s);
        sample.addPos(s);
      }
    }

    
    i =0;
    while(sample.getNegSize()<neg && i<docid.length-1) {
      if(docid.length<100) {
        int randId = RandomUtils.nextInt((int)this.terrier.getTotIndexedDocs());
        String sentId = this.terrier.getDocName(randId);
        String s = FileIO.fileToString(new File(dataDir+sentId));
        if(this.verbose)
          System.out.println("\t\tNeg " + randId + " " + s);
        sample.addNeg(s);
      }
      else {
        i++;
        String sentId = this.terrier.getDocName(docid[(int)(docid.length/1.5)+i]);
        String s = FileIO.fileToString(new File(dataDir+sentId));
        if(!txt.equals(s)) {
          if(this.verbose)
            System.out.println("\t\tNeg " + (i+1) + " " + s);
          sample.addNeg(s);
        }
      }
    }
   return sample;
  }

}
