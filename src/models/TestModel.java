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

import optim.GradientCheck;
import optim.NoiseGradientCalc;

import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.ArrayList;
import java.io.IOException;

public class TestModel {
  public static void main(String[] args) throws IOException {
    Model enModel = new AddModel();
    Model deMdel = new AddModel();

    String file = "sample/english.short";
    String posFile = "sample/english-pos.short";
    String negFile = "sample/english-neg.short";

		String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
//		pipeline.add(PreProcessTerm.SW_REMOVAL);
//		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, Language.EN, path_to_terrier, pipeline);
		Corpus enCorp = new Corpus();
    Dictionary enDict = new Dictionary();
    enCorp.load(file,ch, enDict);


		Channel chPos = new SentFile(posFile);
		chPos.setup(TokenType.WORD, Language.EN, path_to_terrier, pipeline);
		Corpus enPos = new Corpus();
//    Dictionary enDict = new Dictionary();
    enPos.load(posFile,chPos, enDict);

		Channel chNeg = new SentFile(negFile);
		chNeg.setup(TokenType.WORD, Language.EN, path_to_terrier, pipeline);
		Corpus enNeg = new Corpus();
//    Dictionary enDict = new Dictionary();
    enNeg.load(negFile,chNeg, enDict);
    
    System.out.printf("#sentence = %d #tokens = %d\n", enCorp.getSize(), enDict.getSize());
//    enDict.print();

    enModel.setDict(enDict);
    Layer l = new LogisticLayer(64);
    enModel.addHiddenLayer(l);
  
    enModel.init();

/*    for(Sentence s: enCorp.getSentences()) {
      DoubleMatrix rep = enModel.fProp(s);
      rep.print();
    }*/

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

    int batchsize = 1;
    
    for(int i=0; i<instances.size(); i+=2) {
      List<Datum> batch = new ArrayList<Datum>();
      batch.add(instances.get(i));
      batch.add(instances.get(i+1));
      GradientCheck test = new GradientCheck(new NoiseGradientCalc(batch));
      test.optimise(enModel);
    }

  }
}
