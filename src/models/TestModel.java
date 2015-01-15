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
import optim.BasicGradientCalc;
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
    Model deMdel = new AddModel();

    String file = "data/hi-fire-mono/data.txt";
    String posFile = "data/hi-fire-mono/pos-data.txt";
    String negFile = "data/hi-fire-mono/neg-data.txt";

/*  String file = "sample/hindi.short";
    String posFile = "sample/hindi-pos.short";
    String negFile = "sample/hindi-neg.short";*/
		
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
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
    
    System.out.printf("#sentence = %d #tokens = %d\n", enCorp.getSize(), enDict.getSize());
    enDict.save("obj/dict.txt");
//    enDict.print();

    enModel.setDict(enDict);
    Layer l = new LogisticLayer(64);
    enModel.addHiddenLayer(l);
  
    enModel.init();


    List<Corpus> corp = new ArrayList<Corpus>();
    corp.add(enCorp);
    corp.add(enPos);
    corp.add(enNeg);

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

/*    for(int i=0; i<c.get(0).getSize(); i++) {
      List<Sentence> s = new ArrayList<Sentence>();
      for(int corp=0; corp<c.size(); corp++)
        s.add(c.get(corp).get(i));

      gradFunc.setData(s);*/


    
    int batchsize = 100;
    int iterations = 1;

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
/*          GradientCalc trainer = new NoiseGradientCalc(matBatch);
          trainer.setModel(enModel);
          // MAXIMISER
          Optimizer optimizer = new ConjugateGradient(trainer);
          optimizer.optimize(3);
          double[] learntParams = new double[enModel.getThetaSize()];
          trainer.getParameters(learntParams);
          enModel.setParameters(learntParams);
  //        System.out.printf("After Batch %d Cost = %.6f\n", batchNum, trainer.getValue());*/
          batchNum++;
          GradientCheck test = new GradientCheck(new NoiseCosineGradientCalc(matBatch));
          test.optimise(enModel);
          matBatch.close();
        } finally {
          enModel.clearDevice();
        }
         
      }
      enModel.clearDevice();
      enModel.save("obj/model.txt");
    }
  }
}
