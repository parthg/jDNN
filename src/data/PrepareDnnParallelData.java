package data;

import data.PreProcessTerm;
import data.TokenType;
import data.Channel;
import data.SentFile;

import es.upv.nlel.utils.Language;

import common.Dictionary;
import common.Corpus;
import common.Sentence;

import models.Model;
import models.AddModel;

import util.FileUtils;

import math.DMath;
import math.DMatrix;

import java.util.Map;
import java.util.HashMap;
import java.util.Set;
import java.util.HashSet;
import java.util.List;
import java.util.ArrayList;
import java.io.PrintWriter;
import java.io.IOException;
import java.io.File;




public class PrepareDnnParallelData {
  Model enModel, hiModel;
  DMatrix enVocab, hiVocab;
  List<String> enText;
  List<String> hiText;

  // load dict
  public Dictionary loadDict(String file) throws IOException {
    Dictionary dict = new Dictionary();
    dict.load(file);
    return dict;
  }
  
  // load model
  public void loadModel(String enModelFile, Dictionary enDict, String hiModelFile, Dictionary hiDict) throws IOException {
    this.enModel = new AddModel();
    this.enModel.load(enModelFile, enDict);

    this.hiModel = new AddModel();
    this.hiModel.load(hiModelFile, hiDict);
  }

  // project vocab
  public void projectVocabulary() {
    this.enVocab = this.enModel.projectVocabulary(5000);
    this.hiVocab = this.hiModel.projectVocabulary(5000);
  }

  public Corpus loadCorpus(Language l, String file, Channel ch) throws IOException {
    Corpus corp = new Corpus();
    switch(l) {
      case EN: corp.load(file, false, ch, this.enModel.dict(), false);
                        break;
      case HI: corp.load(file, false, ch, this.hiModel.dict(), false);   
                        break;
      default: System.out.printf("Not proper Language options passed %s.\n", l);
               System.exit(0);
    }
    return corp; 
  }

  public DMatrix getRepresentation(Language l, Sentence s) {
    DMatrix dVec = null;
    boolean use = false;
    switch(l) {
      case EN: dVec = DMath.createZerosMatrix(1, this.enModel.outSize());
                        for(int j=0; j<s.getSize(); j++) {
                          use = true;
                          dVec.addi(this.enVocab.getRow(s.get(j)));
                        }
                        break;
      case HI: dVec = DMath.createZerosMatrix(1, this.hiModel.outSize());
                        for(int j=0; j<s.getSize(); j++) {
                          use = true;
                          dVec.addi(this.hiVocab.getRow(s.get(j)));
                        }
                        break;
      default: System.out.printf("Not proper Language options passed %s.\n", l);
               System.exit(0);
    }
    
    if(use)
      return dVec;
    else
      return null;
  }
  
  /** It will us the joint dictionary to create a parallel corpus of type Corpus with certain parameters*/
  public void prepareParallelCorpus(Corpus[] corp, int minLength, int maxSize, String outDir) throws IOException {
    int size = corp[0].getSize();
    assert (size == this.enText.size() && size == this.hiText.size()):System.out.printf("There is an anomaly in corpus index and file lines index.");
    for(int i = 0; i< corp.length; i++) {
      assert (corp[i].getSize() == size):System.out.printf("Both Corpora should have same length.");
    }
    int count = 0;

    PrintWriter pEnRaw = new PrintWriter(outDir+"DNN-subparallel-en.dat", "UTF-8");
    PrintWriter pEnTestRaw = new PrintWriter(outDir+"DNN-subparallel-en-test.dat", "UTF-8");
    
    PrintWriter pHiRaw = new PrintWriter(outDir+"DNN-subparallel-hi.dat", "UTF-8");
    PrintWriter pHiTestRaw = new PrintWriter(outDir + "DNN-subparallel-hi-test.dat", "UTF-8");

    PrintWriter pEn = new PrintWriter(outDir + "DNN-subparallel-projected-en.dat", "UTF-8");
    PrintWriter pEnTest = new PrintWriter(outDir + "DNN-subparallel-projected-en-test.dat", "UTF-8");
    
    PrintWriter pHi = new PrintWriter(outDir + "DNN-subparallel-projected-hi.dat", "UTF-8");
    PrintWriter pHiTest = new PrintWriter(outDir + "DNN-subparallel-projected-hi-test.dat", "UTF-8");

    PrintWriter pEnText = new PrintWriter(outDir + "DNN-subparallel-en-text.txt", "UTF-8");
    PrintWriter pHiText = new PrintWriter(outDir + "DNN-subparallel-hi-text.txt", "UTF-8");

    PrintWriter pEnTestText = new PrintWriter(outDir + "DNN-subparallel-en-test-text.txt", "UTF-8");
    PrintWriter pHiTestText = new PrintWriter(outDir + "DNN-subparallel-hi-test-text.txt", "UTF-8");

    for(int i=0; i<corp[0].getSize(); i++) {
      Sentence s1 = corp[0].get(i);
      Sentence s2 = corp[1].get(i);
      if(s1.getSize()>=minLength && s2.getSize()>=minLength ) {
        if(count < maxSize) {
/*          DMatrix enProj = this.getRepresentation(Language.EN, s1);
          for(int j=0; j< enProj.length(); j++)
            pEn.printf("%.6f ", enProj.get(j));
          pEn.println();*/
          pEnRaw.println(s1.toString());

          DMatrix hiProj = this.getRepresentation(Language.HI, s2);
          for(int j=0; j< hiProj.length(); j++)
            pHi.printf("%.6f ", hiProj.get(j));
          pHi.println();
          pHiRaw.println(s2.toString());

          pEnText.println(this.enText.get(i));
          pHiText.println(this.hiText.get(i));

          count++;
        }
        else {
/*          DMatrix enProj = this.getRepresentation(Language.EN, s1);
          for(int j=0; j< enProj.length(); j++)
            pEnTest.printf("%.6f ", enProj.get(j));
          pEnTest.println();*/
          pEnTestRaw.println(s1.toString());
          
          DMatrix hiProj = this.getRepresentation(Language.HI, s2);
          for(int j=0; j< hiProj.length(); j++)
            pHiTest.printf("%.6f ", hiProj.get(j));
          pHiTest.println();
          pHiTestRaw.println(s2.toString());

          pEnTestText.println(this.enText.get(i));
          pHiTestText.println(this.hiText.get(i));
        }

      }
    }

    pEn.close();
    pEnTest.close();

    pHi.close();
    pHiTest.close();
    
    pEnRaw.close();
    pEnTestRaw.close();

    pHiRaw.close();
    pHiTestRaw.close();

    pEnText.close();
    pHiText.close();

    pEnTestText.close();
    pHiTestText.close();
    System.out.printf("Parallel corpus created with size = %d\n", count);
  }


  public static void main(String[] args) throws IOException {
    PrepareDnnParallelData data = new PrepareDnnParallelData();

    Language lang1 = Language.EN;
    Language lang2 = Language.HI;

    String enDictFile = "data/fire/en/dict-parallel.txt"; // en
    Dictionary enDict = data.loadDict(enDictFile);
    String hiDictFile = "data/fire/hi/dict-titles-full.txt"; // hi
    Dictionary hiDict = data.loadDict(hiDictFile);

    String enFile = "data/fire/en/train.low.eng.sub";
    String hiFile = "data/fire/hi/train.low.hin.sub";

    data.enText = FileUtils.getLines(new File(enFile));
    data.hiText = FileUtils.getLines(new File(hiFile));

//    String enModelFile = "obj/tanh-en-dict-100-b-100-h-128/model_iter27.txt"; // actually this doesn't matter because its not used
//    String hiModelFile = "obj/tanh-hi-dict-400-b-100-h-128-new/model_iter20.txt";

    String enModelFile = "obj/tanh-en-dict-100-b-100-h-128/model_iter27.txt"; // actually this doesn't matter because its not used
    String hiModelFile = "obj/tanh-b-100-h-128/model_iter14.txt";

    data.loadModel(enModelFile, enDict, hiModelFile, hiDict);
    data.projectVocabulary();


    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
    List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
    pipeline.add(PreProcessTerm.SW_REMOVAL);
    pipeline.add(PreProcessTerm.STEM);

    Corpus[] corp = new Corpus[2];
    Channel chEn = new SentFile(enFile);
    chEn.setup(TokenType.WORD, lang1, path_to_terrier, pipeline);
    corp[0] = data.loadCorpus(lang1, enFile, chEn);

    Channel chHi = new SentFile(hiFile);
    chHi.setup(TokenType.WORD, lang2, path_to_terrier, pipeline);
    corp[1] = data.loadCorpus(lang2, hiFile, chHi);

    data.prepareParallelCorpus(corp, 3, 100000, "data/fire/joint-full/");
  }
}
