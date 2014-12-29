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

    System.out.printf("#sentence = %d #tokens = %d\n", enCorp.getSize(), enDict.getSize());
//    enDict.print();

    enModel.setDict(enDict);
    Layer l = new LogisticLayer(5);
    enModel.addHiddenLayer(l);
  
    enModel.init();

    for(Sentence s: enCorp.getSentences()) {
      DoubleMatrix rep = enModel.fProp(s);
      rep.print();
    }

  }
}
