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


		String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
//		pipeline.add(PreProcessTerm.SW_REMOVAL);
//		pipeline.add(PreProcessTerm.STEM);
		
		Channel ch = new SentFile("sample/english");
		ch.setup(TokenType.WORD, Language.EN, path_to_terrier, pipeline);
		Corpus enCorp = new Corpus();
    Dictionary enDict = new Dictionary();
    enCorp.load("sample/english",ch, enDict);


		Channel chPos = new SentFile("sample/english-pos");
		chPos.setup(TokenType.WORD, Language.EN, path_to_terrier, pipeline);
		Corpus enPos = new Corpus();
//    Dictionary enDict = new Dictionary();
    enPos.load("sample/english-pos",chPos, enDict);

		Channel chNeg = new SentFile("sample/english-neg");
		chNeg.setup(TokenType.WORD, Language.EN, path_to_terrier, pipeline);
		Corpus enNeg = new Corpus();
//    Dictionary enDict = new Dictionary();
    enNeg.load("sample/english-neg",chNeg, enDict);
    
    System.out.printf("#sentence = %d #tokens = %d\n", enCorp.getSize(), enDict.getSize());
//    enDict.print();

    enModel.setDict(enDict);
    Layer l = new LogisticLayer(50);
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
    GradientCheck test = new GradientCheck(new NoiseGradientCalc());
    test.optimise(enModel, corp);

  }
}
