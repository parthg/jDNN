package linear;

import java.io.File;
import java.io.PrintWriter;
import java.io.IOException;

import java.util.List;
import java.util.ArrayList;

import common.Dictionary;
import common.Sentence;
import common.Corpus;

import data.PreProcessTerm;
import data.TokenType;
import data.Channel;
import data.SentFile;

import es.upv.nlel.utils.Language;

/** This class would generate the test parallel data (sparse form) for the joint models. Which can later be used by operations like LinearParallelRetrieval.
 * Takse inputs:
 *  1. joint dictionary
 *  2. parallel text files for  both languages
 */
public class LinearTestData {
  Dictionary dict;
  Corpus corp1; 
  Corpus corp2;
  // Load the joint dictionary
  public void loadDict(String file) throws IOException {
    this.dict = new Dictionary();
    this.dict.load(file);
  }
  
  public Corpus loadCorpus(String file, Channel ch) throws IOException {
    Corpus corp = new Corpus();
    corp.load(file, false, ch, this.dict, false);
    return corp; 
  }

  public void prepareTestData(File test1, File test2) throws IOException {
    assert (this.corp1.getSize() == this.corp2.getSize());

    PrintWriter p1 = new PrintWriter(test1);
    PrintWriter p2 = new PrintWriter(test2);

    for(int i=0; i<this.corp1.getSize(); i++) {
      Sentence s1 = corp1.get(i);
      p1.println(s1.toString());

      Sentence s2 = corp2.get(i);
      p2.println(s2.toString());
    }
    p1.close();
    p2.close();
  }


  public static void main(String[] args) throws IOException {
    LinearTestData prepare = new LinearTestData();
    Language lang1 = Language.EN;
    Language lang2 = Language.HI;

    String dict = "data/fire/joint/CL-LSI-dict.txt"; // joint dict

    String enFile = "data/fire/joint/DNN-subparallel-en-test-text.txt";
    String hiFile = "data/fire/joint/DNN-subparallel-hi-test-text.txt";


    prepare.loadDict(dict);

    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
    List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
    pipeline.add(PreProcessTerm.SW_REMOVAL);
    pipeline.add(PreProcessTerm.STEM);

    Channel chEn = new SentFile(enFile);
    chEn.setup(TokenType.WORD, lang1, path_to_terrier, pipeline);
    prepare.corp1 = prepare.loadCorpus(enFile, chEn);

    Channel chHi = new SentFile(hiFile);
    chHi.setup(TokenType.WORD, lang2, path_to_terrier, pipeline);
    prepare.corp2 = prepare.loadCorpus(hiFile, chHi);

    prepare.prepareTestData(new File("data/fire/joint/joint-test-en.dat"), new File("data/fire/joint/joint-test-hi.dat"));
  }
}
