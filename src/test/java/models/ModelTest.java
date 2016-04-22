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

import common.Dictionary;
import common.Sentence;
import common.Corpus;
import models.Model;
import models.AddModel;
import nn.Layer;
import nn.TanhLayer;

public class ModelTest {

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

  @Test
  public void testFProp() throws IOException {
    System.setProperty("representation", "add");
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
    
    Model model = new AddModel();
    model.setDict(dict);
    
    Layer l = new TanhLayer(2);
    model.addHiddenLayer(l);

/*      Layer l2 = new TanhLayer(128);
    enModel.addHiddenLayer(l2);*/

    model.init();
    
    List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);		
    
    Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus corp = new Corpus();
    corp.load(file, false, ch, dict, fillDict);

    double[] params = model.getParameters();
    // 0.1 .. 2.6
    for(int i=0; i<26; i++) {
      params[i] = 0.1*((double)i+1);
    }
    params[26] = -1.0;
    params[27] = -1.5;

    model.setParameters(params);

    assertEquals(62.010, model.weightSquaredSum(), DELTA);
  }
}
