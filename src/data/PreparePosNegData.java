package data;

import data.Channel;
import data.DocCollection;
import data.RandSentences;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;

import java.io.File;
import java.io.PrintWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;

import es.upv.nlel.corpus.NEWSDocType;
import util.CollectionUtils;
import es.upv.nlel.utils.Language;

import es.upv.nlel.wrapper.TerrierWrapper;
import org.terrier.matching.ResultSet;
import es.upv.nlel.utils.FileIO;
import random.RandomUtils;
import data.CleanData;

import antlr.RecognitionException;
import antlr.TokenStreamException;

public class PreparePosNegData {
  NEWSDocType docType;
  Language language;
  // first load titles
  // then load random sentences or probably all sentences
  // treat titles as queries and most relevant sent as pos
  // randomly select neg part or one with least score
  // prepare three such files with line numbers aligned
  List<String> queries;

  public void loadTitleQueriesFromFile(String sentFile) throws IOException {
    this.queries = new ArrayList<String>();
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(sentFile), "UTF-8"));
    String line = "";
    while((line = br.readLine())!=null) {
      this.queries.add(line.trim());
    }
    br.close();
  }
  public void loadTitleQueries(String dir) {
    System.out.printf("Getting titles..\n");
		String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
    
		Channel chTitle = new DocCollection(dir, ".txt");
//		chTitle.setParser(NEWSDocType.TOI)
    chTitle.setParser(this.docType);
    chTitle.setup(TokenType.WORD, this.language, path_to_terrier, pipeline);
		Map<String, Integer> freq = chTitle.getTokensFreq();
    this.queries = chTitle.titles();
    System.out.printf("Total %d titles extracted.\n", this.queries.size());

  }

  public void loadRandomSentences(String dir, String outDir) throws IOException{
		String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
    Channel chRandSent = new RandSentences(dir, ".txt", 2000000);
//		chRandSent.setParser(NEWSDocType.TOI);
    chRandSent.setParser(this.docType);
		chRandSent.setup(TokenType.WORD, this.language, path_to_terrier, pipeline);
		Map<String, Integer> freq = chRandSent.getTokensFreq();
    List<String> sentences = chRandSent.randSentences();

    int id=0;
    for(String s: sentences) {
      PrintWriter p = new PrintWriter(outDir + id+".txt", "UTF-8");
      p.println(s);
      p.close();
      id++;
    }
  }

  public void printDict(int minFreq, TerrierWrapper terrier, String outDir) throws IOException {
    Map<Integer, Integer> tfTable = terrier.getTFTable();
    PrintWriter p = new PrintWriter(outDir+"dict.txt", "UTF-8");
    Set<String> dict = new HashSet<String>();
    int id=0;
    for(int t: tfTable.keySet()) {
      if(tfTable.get(t)>=minFreq) {
        String term = CleanData.parse(terrier.getTerm(t), this.language);
        if(term.length()>0 && !dict.contains(term)) {
          p.printf("%d\t%s\n", id, term);
          dict.add(term);
          id++;
        }
      }
    }
    p.close();
  }

  public static void main(String[] args) throws IOException, RecognitionException, TokenStreamException {
    PreparePosNegData prepare = new PreparePosNegData();
    
    String terrierPath = "/home/parth/workspace/terrier-3.5/";
    String lang = "hi";
    boolean sw_removal = true;
    boolean stem = false;

    if(lang.equals("en"))
      prepare.language = Language.EN;
    else if(lang.equals("hi"))
      prepare.language = Language.HI;

//    String dataDir = "/home/parth/workspace/data/lrec-toi-and-nt/data/toi/2012/";
    String dataDir = "/home/parth/workspace/data/fire/hi.docs.2011/docs/hi_NavbharatTimes/";
    String tempSentDir = "data/fire/"+lang+"/sent/";
    String outDir = "data/fire/"+lang+"/"; // for final files to be written

    prepare.docType = NEWSDocType.NAVBHARAT;


    if(!new File(tempSentDir).exists())
      (new File(tempSentDir)).mkdirs();
    
/*    int existingFiles = (new File(tempSentDir)).list().length;
    if(existingFiles>0) {
      System.out.println("There can not be any file in the tempSentDir, rather there exist " + existingFiles +" files. Please delete them.. Exiting..");
      System.exit(0);
    }*/

    if(new File(outDir + "title-only.txt").exists())
      prepare.loadTitleQueriesFromFile(outDir + "title-only.txt");
    else {
      prepare.loadTitleQueries(dataDir);
      PrintWriter p = new PrintWriter(outDir + "title-only.txt", "UTF-8");
      for(String s: prepare.queries) {
        p.printf("%s\n", s);
      }
      p.close();
    }

//    prepare.loadRandomSentences(dataDir, tempSentDir);

    TerrierWrapper terrier = new TerrierWrapper(terrierPath);


    String indexPath = terrierPath+"var/index/firesent/";
    terrier.setIndex(indexPath, lang);
    terrier.setLanguage(lang);

    if(!new File(indexPath+lang+".docid.map").exists()) {
      System.out.println("Preparing Index...");
      terrier.setIndex(indexPath, lang);
      terrier.prepareIndex(tempSentDir, ".txt", lang, sw_removal, stem);
			terrier.learnDocId(terrierPath +"etc/collection.spec");
    }
    else {
      if(sw_removal)
        terrier.setStopwordRemoval(lang);
      if(stem)
        terrier.setStemmer(lang);
      terrier.learnDocName();
      terrier.loadIndex(indexPath, lang, lang);
    }

    prepare.printDict(5, terrier, outDir);


    System.out.println("Drawing Samples...");

    PrintWriter pData = new PrintWriter(outDir + "data.txt", "UTF-8");
    PrintWriter pPos = new PrintWriter(outDir + "data-pos.txt", "UTF-8");
    PrintWriter pNeg = new PrintWriter(outDir + "data-neg.txt", "UTF-8");

    int counter = 0;
    for(String s: prepare.queries) {
//      if(counter<1000) {
        String q = terrier.pipelineText(s);
        if(q.length()>0) {
          ResultSet rs = terrier.getResultSet(q, "TF_IDF", false, 0);
          int[] docid = rs.getDocids();
          double[] scores = rs.getScores();

          if(docid.length > 0) {

            String sentId = terrier.getDocName(docid[0]);
            String pos = FileIO.fileToString(new File(tempSentDir+sentId)).trim();

            String negSentId = terrier.getDocName(RandomUtils.nextInt((int)terrier.getTotIndexedDocs()));
            String neg = FileIO.fileToString(new File(tempSentDir+negSentId)).trim();
            if(s.length()>10 && pos.length()>10 && neg.length()>10) {
              pData.println(s.replaceAll("\n", " "));
              pPos.println(pos.replaceAll("\n", " "));
              pNeg.println(neg.replaceAll("\n", " "));
              counter++;
            }
          }
        }
//      }
      
    }

    pData.close();
    pPos.close();
    pNeg.close();
    
  }

}
