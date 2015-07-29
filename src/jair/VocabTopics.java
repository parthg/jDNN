package jair;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

import java.io.PrintWriter;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.File;

import es.upv.nlel.wrapper.TerrierWrapper;
import es.upv.nlel.corpus.FIRE;
import es.upv.nlel.corpus.Topic;
import es.upv.nlel.corpus.Topic.Tag;
import es.upv.nlel.utils.Language;

import util.StringUtils;

import data.Channel;
import data.SentFile;
import data.TokenType;
import data.PreProcessTerm;

import common.Dictionary;


public class VocabTopics {
  TerrierWrapper terrier;
  Language qLang;
  List<Topic> topics;
  Dictionary dict;
  Map<String, Double> idf;

  public void loadIdf(String file) throws IOException {
    this.idf = new HashMap<String, Double>();
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
    String line = "";
    while((line = br.readLine())!=null) {
      String[] cols = line.split("\t");
      String term = cols[1].trim();
      Double val = Double.parseDouble(cols[2].trim());
      this.idf.put(term, val);
    }
    br.close();
  }

  public void loadDict(String file) throws IOException {
    this.dict = new Dictionary();
    this.dict.load(file);
  }

  public static void main(String[] args) throws IOException {
    VocabTopics subTopics = new VocabTopics();

    String queryFile = "/home/parth/workspace/data/fire/topics/en.topics.2011-12.txt";
    subTopics.topics = FIRE.parseTopicFile(queryFile);
    subTopics.qLang = Language.EN;

    subTopics.loadDict("data/fire/joint/CL-LSI-dict.txt");
    subTopics.loadIdf("data/fire/joint/joint-idf.txt");
    
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile(null);
		ch.setup(TokenType.WORD, subTopics.qLang, path_to_terrier, pipeline);
    
    int i=0;
    for(Topic t: subTopics.topics) {
      String text = t.get(Tag.TITLE)+" "+t.get(Tag.DESC);
      text = ch.getTokeniser().parse(text);
      text = ch.getTokeniser().clean(text);
      String[] tokens = text.split("_");
      boolean include = false;
      int count = 0;
      for(String tok: tokens) {
        if(subTopics.dict.contains(tok)) {
          if(subTopics.idf.get(tok)>0.0) {
            count ++;
            include = true;
          }
        }
      }
      if(count <tokens.length)
        System.out.printf("%d ", t.getID());
    }
  }
}
