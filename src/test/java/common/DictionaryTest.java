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

public class DictionaryTest {

  static final String dir = "data/test/";
  static final String file = dir+"english";
  static final String posFile = dir+"english-pos";
  static final String negFile = dir+"english-neg";

  static final String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
  static final Language lang = Language.EN;

  static final double DELTA = 0.0000001;

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
  public void testGetBoWRepresentation() throws IOException {
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
		
    List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
    
    Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, lang, path_to_terrier, pipeline);
		Corpus corp = new Corpus();
    corp.load(file, false, ch, dict, fillDict);

    DMatrix vec = dict.getRepresentation(corp.get(1)); // "today is sunday"
    assertArrayEquals(new double[]{0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, vec.data(), DELTA);


//    System.setProperty("use_cuda", "true");
//    assertTrue("Please set use_cuda system property.", Boolean.parseBoolean(System.getProperty("use_cuda")));
/*    DMatrix a = DMath.createOnesMatrix(2,3);
    DMatrix b = DMath.createOnesMatrix(2,3);

    a.muli(2.0);
    assertArrayEquals(new double[]{2.0, 2.0, 2.0, 2.0, 2.0, 2.0}, a.data(), DELTA);

    a = DMath.createMatrix(2, 4, new double[]{0, 1, 2,3, 4, 5, 6, 7});
    b = DMath.createMatrix(2, 4, new double[]{0, 1, 2, 3, 4,5, 6, 7});

    // A.*B
    assertArrayEquals(new double[]{0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0}, a.mul(b).data(), DELTA);*/
  
  }
}
