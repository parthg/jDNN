import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertArrayEquals;

import es.upv.nlel.utils.Language;

import java.util.List;
import java.util.ArrayList;
import java.io.File;
import java.io.IOException;

import math.DMatrix;
import math.DMath;

import data.PreProcessTerm;
import data.Channel;
import data.SentFile;
import data.TokenType;

import common.Datum;
import common.Dictionary;
import common.Sentence;
import common.Corpus;
import common.Batch;

import models.Model;
import models.BoWModel;
import nn.Layer;
import nn.TanhLayer;

import random.RandomUtils;

import optim.GradientCheck;
import optim.GradientCalc;
import optim.NoiseGradientCalc;
import optim.NoiseCosineGradientCalcBoW;

public class NoiseCosineGradientCalcBoWTest {

  static final String dir = "data/test/";
  static final String file = dir+"english";
  static final String posFile = dir+"english-pos";
  static final String negFile = dir+"english-neg";

  static final String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
  static final Language lang = Language.EN;

  static final double DELTA = 0.00001;

  public static void createDictionary() throws IOException {


		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
    
    // ********** DICTIONARY ************* //
    Dictionary dict = new Dictionary();
    boolean fillDict = true;
		
    Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus enCorp = new Corpus();
    enCorp.load(file, false, ch, dict, fillDict);

		Channel chPos = new SentFile(posFile);
		chPos.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus enPos = new Corpus();
    enPos.load(posFile, false, chPos, dict, fillDict);

		Channel chNeg = new SentFile(negFile);
		chNeg.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus enNeg = new Corpus();
    enNeg.load(negFile, false, chNeg, dict, fillDict);

    
    dict.save(dir+"dict.txt");
//    System.out.printf("Total Train Sentences = %d \n", enCorp.getSize());
  }

  public static void createModel() throws IOException {
  }

  @Test 
  public void testGradientCalc() throws IOException {
    Dictionary dict = new Dictionary();
    String dictFile = dir+"dict.txt";
    boolean fillDict = false;
    if(new File(dictFile).exists()) {
      dict.load(dictFile);
    }
    else {
      createDictionary();
    }

    assertEquals(13, dict.getSize());
    
    Model model = new BoWModel();
    model.setDict(dict);
    
    Layer l = new TanhLayer(10);
    model.addHiddenLayer(l);

    model.init();
    
    List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
    Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus enCorp = new Corpus();
    enCorp.load(file, false, ch, dict, fillDict);

		Channel chPos = new SentFile(posFile);
		chPos.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus enPos = new Corpus();
    enPos.load(posFile, false, chPos, dict, fillDict);

		Channel chNeg = new SentFile(negFile);
		chNeg.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus enNeg = new Corpus();
    enNeg.load(negFile, false, chNeg, dict, fillDict);
   
    int[] randArray = new int[enCorp.getSize()];
    for(int i=0; i<randArray.length; i++) {
      randArray[i] = i;
    }

    int count = 0;
    RandomUtils.suffleArray(randArray);
    List<Datum> instances = new ArrayList<Datum>();
    for(int i=0; i<enCorp.getSize(); i++) {
      Sentence s = enCorp.get(randArray[i]);
      Sentence sPos = enPos.get(randArray[i]);
      Sentence sNeg = enNeg.get(randArray[i]);
      List<Sentence> nSents = new ArrayList<Sentence>();
      nSents.add(sNeg);
      if(s.getSize()>0 && sPos.getSize()>0 && sNeg.getSize()>0) {
        Datum d = new Datum(count, s, sPos, nSents);
        instances.add(d);
        count++;
      }
    }

    try(Batch matBatch = new Batch(instances, 1, model.dict());) {
      matBatch.copyHtoD();
/*        GradientCalc trainer = new NoiseCosineGradientCalcBoW(matBatch);
      trainer.setModel(model);
      // MAXIMISER
      Optimizer optimizer = new ConjugateGradient(trainer);
      optimizer.optimize(3);
      double[] learntParams = new double[enModel.getThetaSize()];
      trainer.getParameters(learntParams);
      enModel.setParameters(learntParams);
      batchNum++;*/
      GradientCheck gCheck = new GradientCheck(new NoiseCosineGradientCalcBoW(matBatch));
      gCheck.optimise(model);
    }
  }

}
