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
  Model lang1Model, lang2Model;
  DMatrix lang1Vocab, lang2Vocab;
  List<String> lang1Text;
  List<String> lang2Text;
  Language lang1, lang2;

  // load dict
  public Dictionary loadDict(String file) throws IOException {
    Dictionary dict = new Dictionary();
    dict.load(file);
    return dict;
  }
  
  // load model
  public void loadModel(String lang1ModelFile, Dictionary lang1Dict, String lang2ModelFile, Dictionary lang2Dict) throws IOException {
    this.lang1Model = new AddModel();
    this.lang1Model.setDict(lang1Dict);
//    this.enModel.load(enModelFile, enDict);

    this.lang2Model = new AddModel();
    this.lang2Model.load(lang2ModelFile, lang2Dict);
  }

  // project vocab
  public void projectVocabulary() {
//    this.enVocab = this.enModel.projectVocabulary(5000);
    this.lang2Vocab = this.lang2Model.projectVocabulary(5000);
  }

  public Corpus loadCorpus(int l, String file, Channel ch) throws IOException {
    Corpus corp = new Corpus();
    switch(l) {
      case 1: corp.load(file, false, ch, this.lang1Model.dict(), false);
                        break;
      case 2: corp.load(file, false, ch, this.lang2Model.dict(), false);   
                        break;
/*      case ES: corp.load(file, false, ch, this.lang1Model.dict(), false);   
                        break;*/
      default: System.out.printf("Not proper Language options passed %d (it can be either 1 or 2).\n", l);
               System.exit(0);
    }
    return corp; 
  }

  public DMatrix getRepresentation(int l, Sentence s) {
    DMatrix dVec = null;
    boolean use = false;
    switch(l) {
      case 1: dVec = DMath.createZerosMatrix(1, this.lang1Model.outSize());
                        for(int j=0; j<s.getSize(); j++) {
                          use = true;
                          dVec.addi(this.lang1Vocab.getRow(s.get(j)));
                        }
                        break;
      case 2: dVec = DMath.createZerosMatrix(1, this.lang2Model.outSize());
                        for(int j=0; j<s.getSize(); j++) {
                          use = true;
                          dVec.addi(this.lang2Vocab.getRow(s.get(j)));
                        }
                        break;
      default: System.out.printf("Not proper Language options passed %d (it can be either 1 or 2).\n", l);
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
    assert (size == this.lang1Text.size() && size == this.lang2Text.size()):System.out.printf("There is an anomaly in corpus index and file lines index.");
    for(int i = 0; i< corp.length; i++) {
      assert (corp[i].getSize() == size):System.out.printf("Both Corpora should have same length.");
    }
    int count = 0;

    PrintWriter pEnRaw = new PrintWriter(outDir+"DNN-subparallel-"+this.lang1.getCode()+".dat", "UTF-8");
    PrintWriter pEnTestRaw = new PrintWriter(outDir+"DNN-subparallel-"+this.lang1.getCode()+"-test.dat", "UTF-8");
    
    PrintWriter pHiRaw = new PrintWriter(outDir+"DNN-subparallel-"+this.lang2.getCode()+".dat", "UTF-8");
    PrintWriter pHiTestRaw = new PrintWriter(outDir + "DNN-subparallel-"+this.lang2.getCode()+"-test.dat", "UTF-8");

    PrintWriter pEn = new PrintWriter(outDir + "DNN-subparallel-projected-"+this.lang1.getCode()+".dat", "UTF-8");
    PrintWriter pEnTest = new PrintWriter(outDir + "DNN-subparallel-projected-"+this.lang1.getCode()+"-test.dat", "UTF-8");
    
    PrintWriter pHi = new PrintWriter(outDir + "DNN-subparallel-projected-"+this.lang2.getCode()+".dat", "UTF-8");
    PrintWriter pHiTest = new PrintWriter(outDir + "DNN-subparallel-projected-"+this.lang2.getCode()+"-test.dat", "UTF-8");

    PrintWriter pEnText = new PrintWriter(outDir + "DNN-subparallel-"+this.lang1.getCode()+"-text.txt", "UTF-8");
    PrintWriter pHiText = new PrintWriter(outDir + "DNN-subparallel-"+this.lang2.getCode()+"-text.txt", "UTF-8");

    PrintWriter pEnTestText = new PrintWriter(outDir + "DNN-subparallel-"+this.lang1.getCode()+"-test-text.txt", "UTF-8");
    PrintWriter pHiTestText = new PrintWriter(outDir + "DNN-subparallel-"+this.lang2.getCode()+"-test-text.txt", "UTF-8");

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

          DMatrix lang2Proj = this.getRepresentation(2, s2);
          for(int j=0; j< lang2Proj.length(); j++)
            pHi.printf("%.6f ", lang2Proj.get(j));
          pHi.println();
          pHiRaw.println(s2.toString());

          pEnText.println(this.lang1Text.get(i));
          pHiText.println(this.lang2Text.get(i));

          count++;
        }
        else {
/*          DMatrix enProj = this.getRepresentation(Language.EN, s1);
          for(int j=0; j< enProj.length(); j++)
            pEnTest.printf("%.6f ", enProj.get(j));
          pEnTest.println();*/
          pEnTestRaw.println(s1.toString());
          
          DMatrix lang2Proj = this.getRepresentation(2, s2);
          for(int j=0; j< lang2Proj.length(); j++)
            pHiTest.printf("%.6f ", lang2Proj.get(j));
          pHiTest.println();
          pHiTestRaw.println(s2.toString());

          pEnTestText.println(this.lang1Text.get(i));
          pHiTestText.println(this.lang2Text.get(i));
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

    String corpus = "clef"; // "fire-new" OR "clef"
    data.lang1 = Language.DE; // query lang
    data.lang2 = Language.EN; // doc lang

//    String enDictFile = "data/fire/en/dict-parallel.txt"; // en
    String lang1DictFile = "data/"+corpus+"/"+data.lang1.getCode()+"/dict-top10000.txt"; // en
    Dictionary lang1Dict = data.loadDict(lang1DictFile);
//    String hiDictFile = "data/fire/hi/dict-titles-full.txt"; // hi
    String lang2DictFile = "data/"+corpus+"/"+data.lang2.getCode()+"/dict-top10000.txt"; // hi
    Dictionary lang2Dict = data.loadDict(lang2DictFile);

//    String lang1File = "data/"+corpus+"/"+data.lang1.getCode()+"/train.low.eng.sub";
//    String lang2File = "data/"+corpus+"/"+data.lang2.getCode()+"/train.low.hin.sub";
    String lang1File = "data/clef/de/de-parallel-corpus.txt";
    String lang2File = "data/clef/en/en-parallel-corpus.txt";

    data.lang1Text = FileUtils.getLines(new File(lang1File));
    data.lang2Text = FileUtils.getLines(new File(lang2File));

//    String enModelFile = "obj/tanh-en-dict-100-b-100-h-128/model_iter27.txt"; // actually this doesn't matter because its not used
//    String hiModelFile = "obj/tanh-hi-dict-400-b-100-h-128-new/model_iter20.txt";

//    String enModelFile = "obj/tanh-en-dict-100-b-100-h-128/model_iter27.txt"; // actually this doesn't matter because its not used
//    String hiModelFile = "obj/tanh-b-100-h-128/model_iter14.txt";

    String lang1ModelFile = "obj/tanh-en-dict-100-b-100-h-128/model_iter27.txt"; // actually this doesn't matter because its not used
//    String lang2ModelFile = "obj/tanh-es-dict-top-10k-b-200-h-128/model_iter29.txt";
//    String lang2ModelFile = "obj/tanh-clef-en-w-0.5-10k-b-100-h-128-bias-1.0/model_iter8.txt";
//    String lang2ModelFile = "obj/tanh-fire-hi-w-0.5-10k-b-100-h-128-bias-1.0/model_iter5.txt";
    String lang2ModelFile = "obj/tanh-clef-en-w-0.5-10k-b-100-h-128-bias-1.0-new/model_iter6.txt";
    
    data.loadModel(lang1ModelFile, lang1Dict, lang2ModelFile, lang2Dict);
    data.projectVocabulary();


    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
    List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
    pipeline.add(PreProcessTerm.SW_REMOVAL);
    pipeline.add(PreProcessTerm.STEM);

    Corpus[] corp = new Corpus[2];
    Channel chLang1 = new SentFile(lang1File);
    chLang1.setup(TokenType.WORD, data.lang1, path_to_terrier, pipeline);
    corp[0] = data.loadCorpus(1, lang1File, chLang1);

    Channel chLang2 = new SentFile(lang2File);
    chLang2.setup(TokenType.WORD, data.lang2, path_to_terrier, pipeline);
    corp[1] = data.loadCorpus(2, lang2File, chLang2);

    data.prepareParallelCorpus(corp, 3, 250000, "data/"+corpus+"/joint-de/"); // 12500 for fire and 250000 for clef
  }
}
