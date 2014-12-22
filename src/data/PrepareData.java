package data;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import es.upv.nlel.corpus.NEWSDocType;
import util.CollectionUtils;
import es.upv.nlel.utils.Language;

public class PrepareData {
	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException  {
		String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		List<PreProcessTerm> pipeline = new ArrayList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
		
		Channel chParallel = new SentFile("/home/parth/workspace/data/wmt-qt/hi-raw.txt");
		chParallel.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
		Map<String, Integer> freq = chParallel.getTokensFreq();
		
		Channel chTitle = new DocCollection("/home/parth/workspace/data/fire/hi.docs.2011/docs/hi_NavbharatTimes", ".txt");
		chTitle.setParser(NEWSDocType.NAVBHARAT);
		chTitle.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
		Map<String, Integer> freq2 = chTitle.getTokensFreq();
		CollectionUtils.aggregateMaps(freq, freq2);
		
		Channel chRandSent = new RandSentences("/home/parth/workspace/data/fire/hi.docs.2011/docs/hi_NavbharatTimes", ".txt", 100000);
		chRandSent.setParser(NEWSDocType.NAVBHARAT);
		chRandSent.setup(TokenType.WORD, Language.HI, path_to_terrier, pipeline);
		Map<String, Integer> freq3 = chRandSent.getTokensFreq();
		
		
		CollectionUtils.aggregateMaps(freq, freq3);
		System.out.println(freq.size());
		
		Map<String, Integer> tokenIndex = getTokenIndex(freq, 2 );
		chParallel.setTokensIndex(tokenIndex);
		chTitle.setTokensIndex(tokenIndex);
		chRandSent.setTokensIndex(tokenIndex);
		
		CollectionUtils.printMap(tokenIndex, new File("etc/data/fire/hi/term-index.txt"));
		
		System.out.println(tokenIndex.size());
		
		Map<Integer, Map<Integer, Integer>> m1 = chParallel.getMatrix();
		System.out.println("Size of data 1 = " + m1.size());
		Map<Integer, Map<Integer, Integer>> m2 = chTitle.getMatrix();
		System.out.println("Size of data 2 = " + m2.size());
		Map<Integer, Map<Integer, Integer>> m3 = chRandSent.getMatrix();
		System.out.println("Size of data 3 = " + m3.size());
		
		CollectionUtils.appendMaps(m1, m2);
		CollectionUtils.appendMaps(m1, m3);
		
		System.out.println("After appending size of data 1 = " + m1.size());
		
		CollectionUtils.printMapOfMap(m1, new File("etc/data/fire/hi/data-matrix.txt"));
		
		Map<Integer, String> data1 = chParallel.getDataIndex();
		System.out.println("Size of index 1 = " + data1.size());
		Map<Integer, String> data2 = chTitle.getDataIndex();
		System.out.println("Size of index 2 = " + data2.size());
		Map<Integer, String> data3 = chRandSent.getDataIndex();
		System.out.println("Size of index 3 = " + data3.size());
		
		CollectionUtils.appendMaps(data1, data2);
		CollectionUtils.appendMaps(data1, data3);
		
		System.out.println("After appending size of index 1 = " + data1.size());
		
		CollectionUtils.printMap(data1, new File("etc/data/fire/hi/doc-index.txt"));
		
		
		// get matrices from all of them
		
		// get data id info back from channels and stor them properly
	}
	
	
	
	static Map<String, Integer> getTokenIndex(Map<String, Integer> freq, int theta) {
		Map<String, Integer> tokenIndex = new HashMap<String, Integer>();
		int id = 0;
		for(String tok: freq.keySet()) {
			if(freq.get(tok)>=theta) {
				tokenIndex.put(tok, id);
				id++;
			}
		}
		return tokenIndex;
	}
	
}
