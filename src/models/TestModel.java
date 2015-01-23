package models;

import models.Model;

import models.AddModel;
import data.PreProcessTerm;
import data.Channel;
import data.SentFile;
import data.TokenType;

import nn.Layer;
import nn.LogisticLayer;

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

import cc.mallet.optimize.ConjugateGradient;
import cc.mallet.optimize.Optimizer;

import java.util.List;
import java.util.ArrayList;
import java.io.IOException;

public class TestModel {
  public static void main(String[] args) throws IOException {
    Model enModel = new AddModel();
    Model hiModel = new AddModel();

    boolean test = true;



    String file = "data/hi-fire-mono/data.txt";
    String posFile = "data/hi-fire-mono/pos-data.txt";
    String negFile = "data/hi-fire-mono/neg-data.txt";

    String test_file = "data/hi-fire-mono/data-test.txt";
    String test_posFile = "data/hi-fire-mono/pos-data-test.txt";
    String test_negFile = "data/hi-fire-mono/neg-data-test.txt";

/*  String file = "sample/hindi.short";
    String posFile = "sample/hindi-pos.short";
    String negFile = "sample/hindi-neg.short";*/
		
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
    // *******  TRAIN *************
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
		Corpus enCorp = new Corpus();
    Dictionary enDict = new Dictionary();
    enCorp.load(file,ch, enDict);

    pipeline.remove(PreProcessTerm.SW_REMOVAL);
    pipeline.remove(PreProcessTerm.STEM);

		Channel chPos = new SentFile(posFile);
		chPos.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
		Corpus enPos = new Corpus();
//    Dictionary enDict = new Dictionary();
    enPos.load(posFile,chPos, enDict);

		Channel chNeg = new SentFile(negFile);
		chNeg.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
		Corpus enNeg = new Corpus();
//    Dictionary enDict = new Dictionary();
    enNeg.load(negFile,chNeg, enDict);

    System.out.printf("Total Train Sentences = %d \n", enCorp.getSize());

    //********  TEST ***************//
    Channel chTest = new SentFile(test_file);
    Corpus enTest = new Corpus();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
    if(test) {
      chTest.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
      enTest.load(test_file, chTest, enDict);
    }

    // TODO: If you have to change the pipeline, do it here
		pipeline.remove(PreProcessTerm.SW_REMOVAL);
		pipeline.remove(PreProcessTerm.STEM);
    chTest = new SentFile(test_posFile);
    Corpus enTestPos = new Corpus();
    if(test) {
      chTest.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
      enTestPos.load(test_posFile, chTest, enDict);
    }
    
    chTest = new SentFile(test_negFile);
    Corpus enTestNeg = new Corpus();
    if(test) {
      chTest.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
      enTestNeg.load(test_negFile, chTest, enDict);
    }
    System.out.printf("Total Test Sentences = %d %d %d \n", enTest.getSize(), enTestPos.getSize(), enTestNeg.getSize());
    
    System.out.printf("#sentence = %d #tokens = %d\n", enCorp.getSize(), enDict.getSize());
    enDict.save("obj/dict.txt");
//    enDict.print();

    enModel.setDict(enDict);
    Layer l = new LogisticLayer(64);
    enModel.addHiddenLayer(l);
  
    enModel.init(1.0, 0.0);

    List<Datum> instances = new ArrayList<Datum>();
    for(int i=0; i<enCorp.getSize(); i++) {
      Sentence s = enCorp.get(i);
      Sentence sPos = enPos.get(i);
      Sentence sNeg = enNeg.get(i);
      List<Sentence> nSents = new ArrayList<Sentence>();
      nSents.add(sNeg);
      Datum d = new Datum(i, s, sPos, nSents);
      instances.add(d);
    }

    List<Datum> test_instances = new ArrayList<Datum>();
    for(int i=0; i<enTest.getSize(); i++) {
      Sentence s = enTest.get(i);
      Sentence sPos = enTestPos.get(i);
      Sentence sNeg = enTestNeg.get(i);
      List<Sentence> nSents = new ArrayList<Sentence>();
      nSents.add(sNeg);
      Datum d = new Datum(i, s, sPos, nSents);
      test_instances.add(d);
    }

    try(Batch testBatch = new Batch(test_instances, 1, enModel.dict())) {
      if(test)
        testBatch.copyHtoD();
      
      int batchsize = 100;
      int iterations = 10;

      for(int iter = 0; iter<iterations; iter++) {
        int batchNum = 1;
        System.out.printf("\n\nIteration = %d", iter+1);
        for(int i=0; i<instances.size(); i+=batchsize) {
          int innerbatchsize = batchsize;
          System.out.printf("\n\n\nBatch = %d\n", batchNum);
          int left = instances.size()-i;
          if(left<batchsize)
            innerbatchsize=left;
          List<Datum> batch = new ArrayList<Datum>();
          for(int j=0; j<innerbatchsize; j++) {
            batch.add(instances.get(i+j));
          } 
          try(Batch matBatch = new Batch(batch, 1, enModel.dict());) {
            matBatch.copyHtoD();
/*            GradientCalc trainer = new NoiseGradientCalc(matBatch);
            trainer.setModel(enModel);
            // MAXIMISER
            Optimizer optimizer = new ConjugateGradient(trainer);
            optimizer.optimize(3);
            double[] learntParams = new double[enModel.getThetaSize()];
            trainer.getParameters(learntParams);
            enModel.setParameters(learntParams);
            if(test) {
              trainer.testStats(testBatch);
              System.out.printf("After Batch %d Cost = %.6f and MRR = %.6f\n", batchNum, trainer.testLoss(), trainer.testMRR());
            }*/
            batchNum++;
            GradientCheck gCheck = new GradientCheck(new NoiseCosineGradientCalc(matBatch));
            gCheck.optimise(enModel);
            matBatch.close();
          } finally {
            enModel.clearDevice();
          }
           
        }
        enModel.clearDevice();
        enModel.save("obj/model_iter"+iter+".txt");
      } // for iterations closed

      testBatch.close();
    }// try test closed
  } // main closed
}
