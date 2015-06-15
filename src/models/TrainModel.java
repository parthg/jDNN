package models;

import models.Model;

import models.AddModel;
import data.PreProcessTerm;
import data.Channel;
import data.SentFile;
import data.TokenType;

import nn.Layer;
import nn.LogisticLayer;
import nn.TanhLayer;

import es.upv.nlel.utils.Language;

import common.Sentence;
import common.Corpus;
import common.Dictionary;
import common.Datum;
import common.Batch;

import optim.GradientCheck;
import optim.GradientCalc;
//import optim.BasicGradientCalc;
import optim.NoiseGradientCalc;
import optim.NoiseCosineGradientCalc;
import optim.NoiseCosineGradientCalcDeep;


import math.jcublas.SimpleCuBlas;

import cc.mallet.optimize.ConjugateGradient;
import cc.mallet.optimize.Optimizer;

import random.RandomUtils;

import java.util.List;
import java.util.ArrayList;
import java.io.IOException;
import java.io.File;

public class TrainModel {
  public static void main(String[] args) throws IOException {
    if(args.length!=1) {
      System.out.printf("Usage: sh run.sh models.TrainModel <prefix>\n");
      System.exit(0);
    }
    Model enModel = new AddModel();
   
    Language lang = Language.ES;
    String prefix = args[0];
    boolean test = true; // will skip or not the test module
    boolean randomize = true; // should always be true if you don't have a reason
    boolean trainContinue=false;	// it is to continue training the model but the mini-batches will change
    int lastIter = 0;	// if Train continue, then specify from what iteration you want to continue
    boolean fillDict = true;	// if true, it prepares the dictionary from the data
    String modelFile = "obj/tanh-hi-dict-400-b-100-h-128-new/model_iter33.txt";
//    String dictFile = "obj/"+prefix+"/dict.txt";
    String useDict = "data/fire/"+lang.getCode()+"/dict-top10000.txt";

    String modelDir = "obj/"+prefix+"/";
    if(!new File(modelDir).exists())
      new File(modelDir).mkdirs();

/*    String file = "data/hi-fire-mono/data.txt";
    String posFile = "data/hi-fire-mono/pos-data.txt";
    String negFile = "data/hi-fire-mono/neg-data.txt";

    String test_file = "data/hi-fire-mono/data-test.txt";
    String test_posFile = "data/hi-fire-mono/pos-data-test.txt";
    String test_negFile = "data/hi-fire-mono/neg-data-test.txt";*/

    String file = "data/fire/"+lang.getCode()+"/data.txt";
    String posFile = "data/fire/"+lang.getCode()+"/data-pos.txt";
    String negFile = "data/fire/"+lang.getCode()+"/data-neg.txt";

    String test_file = "data/fire/"+lang.getCode()+"/data-test.txt";
    String test_posFile = "data/fire/"+lang.getCode()+"/data-pos-test.txt";
    String test_negFile = "data/fire/"+lang.getCode()+"/data-neg-test.txt";

/*    String file = "sample/hindi.short";
    String posFile = "sample/hindi-pos.short";
    String negFile = "sample/hindi-neg.short";*/
		
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
    
    // ********** DICTIONARY ************* //
    Dictionary enDict = new Dictionary();
    if(trainContinue || useDict.length()>0) {
      enDict.load(useDict);
      fillDict = false;
    }

    // *******  TRAIN DATA ************* //
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus enCorp = new Corpus();
    enCorp.load(file, false, ch, enDict, fillDict);

/*    pipeline.remove(PreProcessTerm.SW_REMOVAL);
    pipeline.remove(PreProcessTerm.STEM);*/

		Channel chPos = new SentFile(posFile);
		chPos.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus enPos = new Corpus();
    enPos.load(posFile, false, chPos, enDict, fillDict);

		Channel chNeg = new SentFile(negFile);
		chNeg.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus enNeg = new Corpus();
    enNeg.load(negFile, false, chNeg, enDict, fillDict);

    System.out.printf("Total Train Sentences = %d \n", enCorp.getSize());

    // ********  TEST DATA *************** //
    Channel chTest = new SentFile(test_file);
    Corpus enTest = new Corpus();
/*		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);*/
    if(test) {
      chTest.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
      enTest.load(test_file, false, chTest, enDict, fillDict);
    }

/*		pipeline.remove(PreProcessTerm.SW_REMOVAL);
		pipeline.remove(PreProcessTerm.STEM);*/
    chTest = new SentFile(test_posFile);
    Corpus enTestPos = new Corpus();
    if(test) {
      chTest.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
      enTestPos.load(test_posFile, false, chTest, enDict, fillDict);
    }
    
    chTest = new SentFile(test_negFile);
    Corpus enTestNeg = new Corpus();
    if(test) {
      chTest.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
      enTestNeg.load(test_negFile, false, chTest, enDict, fillDict);
    }
    System.out.printf("Total Test Sentences = %d %d %d \n", enTest.getSize(), enTestPos.getSize(), enTestNeg.getSize());
    
    System.out.printf("#sentence = %d #tokens = %d\n", enCorp.getSize(), enDict.getSize());
    

	/******* MODEL ********/
    if(trainContinue) {
      enModel.load(modelFile, enDict);
/*      double[] params = enModel.getParameters();
      for(int i=0; i<params.length; i++)
        params[i] = 0.1*params[i];
      enModel.setParameters(params);*/
      System.out.printf("Model loaded to continue training from iteration = %d\n", (lastIter+1));
    }
    else {
      enModel.setDict(enDict);
      enDict.save(modelDir+"dict.txt");
      Layer l = new TanhLayer(128);
      enModel.addHiddenLayer(l);

/*      Layer l2 = new TanhLayer(128);
      enModel.addHiddenLayer(l2);*/
  
      enModel.init(1.0, 0.0);
      enModel.printArchitecture();
    }
    int[] randArray = new int[enCorp.getSize()];
    for(int i=0; i<enCorp.getSize(); i++)
      randArray[i] = i;
    
    if(randomize)
      RandomUtils.suffleArray(randArray);

	/***** Train Instances ******/
    List<Datum> instances = new ArrayList<Datum>();
    int count = 0;
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
	
	/***** TEST INSTANCES **********/
    List<Datum> test_instances = new ArrayList<Datum>();
    count = 0;
    for(int i=0; i<enTest.getSize(); i++) {
      Sentence s = enTest.get(i);
      Sentence sPos = enTestPos.get(i);
      Sentence sNeg = enTestNeg.get(i);
      List<Sentence> nSents = new ArrayList<Sentence>();
      nSents.add(sNeg);
      if(s.getSize()>0 && sPos.getSize()>0 && sNeg.getSize()>0) {
        Datum d = new Datum(count, s, sPos, nSents);
        test_instances.add(d);
        count++;
      }
    }

    System.out.printf("Finally, train instances = %d and test instances = %d\n", instances.size(), test_instances.size());




    try(Batch testBatch = new Batch(test_instances, 1, enModel.dict())) {
      
      int batchsize = 200;
      int iterations = 100;

      for(int iter = lastIter+1; iter<=iterations; iter++) {
        int batchNum = 1;
        if(randomize)
          RandomUtils.suffleArray(randArray);

        instances = new ArrayList<Datum>();
        count = 0;
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

        System.out.printf("\n\nIteration = %d", iter);
        for(int i=0; i<instances.size(); i+=batchsize) {
          int innerbatchsize = batchsize;
          System.out.printf("\nBatch = %d ", batchNum);
          int left = instances.size()-i;
          if(left<batchsize)
            innerbatchsize=left;
          List<Datum> batch = new ArrayList<Datum>();
          for(int j=0; j<innerbatchsize; j++) {
            batch.add(instances.get(i+j));
          } 
          try(Batch matBatch = new Batch(batch, 1, enModel.dict());) {
            matBatch.copyHtoD();
            GradientCalc trainer = new NoiseCosineGradientCalc(matBatch);
            trainer.setModel(enModel);
            // MAXIMISER
            Optimizer optimizer = new ConjugateGradient(trainer);
            optimizer.optimize(1);
            double[] learntParams = new double[enModel.getThetaSize()];
            trainer.getParameters(learntParams);
            enModel.setParameters(learntParams);
            batchNum++;
/*            GradientCheck gCheck = new GradientCheck(new NoiseCosineGradientCalc(matBatch));
            gCheck.optimise(enModel);*/
            matBatch.close();
            if(batchNum%100==0) {
              trainer.testStats(testBatch);
              System.out.printf("\nAfter Batch %d Test Cost = %.6f and Test MRR = %.6f\n\n", batchNum, trainer.testLoss(), trainer.testMRR());
            }
            if(SimpleCuBlas.cudaCount > 0)
              System.out.printf("At end of batch cudaCount = %d\n", SimpleCuBlas.cudaCount);

            SimpleCuBlas.reset();
          } finally {
            enModel.clearDevice();
          }
           
        }
        if(test) {
          GradientCalc trainer = new NoiseCosineGradientCalc(null);
          trainer.setModel(enModel);
          trainer.testStats(testBatch);
          System.out.printf("After Iteration %d Cost = %.6f and MRR = %.6f\n\n", (iter), trainer.testLoss(), trainer.testMRR());
        }
        enModel.clearDevice();
        enModel.save(modelDir+"model_iter"+iter+".txt");
        SimpleCuBlas.reset();
      } // for iterations closed

      testBatch.close();
      SimpleCuBlas.reset();
    }// try test closed
  } // main closed
}
