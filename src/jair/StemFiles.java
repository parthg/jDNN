package jair;

import java.util.List;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.File;

import es.upv.nlel.corpus.FIRE;
import es.upv.nlel.corpus.Topic;
import es.upv.nlel.corpus.Topic.Tag;
import es.upv.nlel.utils.Language;
import es.upv.nlel.wrapper.TerrierWrapper;
import es.upv.nlel.preprocess.Stemmer;
import es.upv.nlel.preprocess.HindiStemmerLight;

public class StemFiles {
  TerrierWrapper terrier;
  Stemmer stemmer;
  Language lang;

  public static void main(String[] args) throws IOException {
    String dataFile = "etc/data/fire/hi/title-only-with-did.txt";
    String outFile = "etc/data/fire/hi/title-only-with-did-stemmed.txt";
    StemFiles stemData = new StemFiles();
    stemData.setStemmer(Language.HI);
    stemData.stemSentenceCollection(dataFile, outFile);

    String queryFile = "/home/parth/workspace/data/fire/topics/en.topics.126-175.2011.txt";
    String outQueryFile = "/home/parth/workspace/data/fire/topics/en.topics.126-175.2011.stemmed.txt";
    stemData.setStemmer(Language.EN);
    stemData.stemQueryFile(queryFile, outQueryFile);
  }

  public void stemQueryFile(String inFile, String outFile) throws IOException {
    List<Topic> topics = FIRE.parseTopicFile(inFile);
    PrintWriter p = new PrintWriter(outFile, "UTF-8");
    p.println("<topics>\n");
    for(Topic t: topics) {
      p.println("<top>");
      p.println("<num>"+t.getID()+"</num>");
      p.println("<title>"+this.stemString(t.get(Tag.TITLE))+"</title>");
      p.println("<desc>"+this.stemString(t.get(Tag.DESC))+"</desc>");
      p.println("<narr>"+this.stemString(t.get(Tag.NARR))+"</narr>");
      p.println("</top>\n");
    }
    p.println("</topics>");
    p.close();
  }

  public void stemSentenceCollection(String inFile, String outFile) throws IOException {
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(inFile)),"UTF-8"));
    PrintWriter p = new PrintWriter(outFile, "UTF-8");
    String line = "";
    while((line = br.readLine())!=null) {
      String[] cols = line.split("\t");
      String id = cols[0].trim();
      String newLine = this.stemString(cols[1].trim());
      p.println(id+"\t"+newLine);
    }
    br.close();
    p.close();
  }

  public void setStemmer(Language _lang) {
    this.lang = _lang;
		this.terrier = new TerrierWrapper("/home/parth/workspace/terrier-3.5");
		this.stemmer = null;
		if(this.lang == Language.EN) {
			this.terrier.setLanguage("en");
			this.terrier.setStemmer("en");
		}
		else if(this.lang == Language.HI)
			this.stemmer = new HindiStemmerLight();
/*		else if(this.lang == Language.ES) {
			this.terrier.setLanguage("es");
			this.terrier.setStemmer("es");
		}*/
  }

  public String stemString(String line) {
    String newLine = "";
    String[] terms = line.split(" ");
    for(String s : terms) {
      String stem = "";
      if(this.lang == Language.EN) {
        try {
          stem = this.terrier.stem(s);
        }
        catch(Exception e) {
          stem = s;
        }
      }
      else if(this.lang == Language.HI)
        stem = this.stemmer.stem(s);
      else
        stem = s;
      
      newLine += (" "+ stem);
    }
    return newLine.trim();
  }
}
