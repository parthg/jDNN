package jair;

import java.util.List;
import java.util.ArrayList;
import java.io.IOException;

import data.PreProcessTerm;
import data.Channel;
import data.SentFile;
import data.TokenType;


import es.upv.nlel.utils.Language;

import common.Corpus;
import common.Dictionary;


public class PrepareDictionary {
  public static void main(String[] args) throws IOException {
    String file = "data/fire/joint/DNN-subparallel-en-text.txt";
    String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
//		pipeline.add(PreProcessTerm.STEM);
		
    
    // ********** DICTIONARY ************* //
    Dictionary enDict = new Dictionary();

    // *******  TRAIN ************* //
		Channel ch = new SentFile(file);
		ch.setup(TokenType.WORD, Language.EN, path_to_terrier, pipeline);
		Corpus enCorp = new Corpus();
    enCorp.load(file, false, ch, enDict, true);

    enDict.save("data/fire/en/dict-parallel.txt");
  }
}
