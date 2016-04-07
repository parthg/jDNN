package data;

import java.io.IOException;
import java.util.List;
import java.util.LinkedList;
import java.lang.StringBuilder;

import es.upv.nlel.utils.Language;
import es.upv.nlel.wrapper.TerrierWrapper;

public class WordHashTokeniser implements Tokeniser {
	Language lang;
	TerrierWrapper terrier;
	List<PreProcessTerm> termPipeline;
	
	public void setup(Language _lang, String path_to_terrier, List<PreProcessTerm> pipeline) {
		this.lang = _lang;
		this.terrier = new TerrierWrapper(path_to_terrier);
		this.termPipeline = pipeline;
		
		this.terrier.setLanguage(_lang.getCode());
		for(PreProcessTerm step: this.termPipeline) {
			switch(step) {
			case SW_REMOVAL: 
				this.terrier.setStopwordRemoval(this.lang.getCode());
				break;
			case STEM:
				this.terrier.setStemmer(this.lang.getCode());
				break;
			default:
				System.out.println("Not a proper Step..");
				break;
			}
		}
	}

  public static List<String> getCharNGrams(String str, int n) {
    List<String> grams = new LinkedList<String>();
    char[] chars = str.toCharArray();
    for(int i=0; i<=str.length()-n; i++) {
      StringBuilder sb = new StringBuilder();
      for(int j=0; j<n; j++)
        sb.append(chars[i+j]);
      grams.add(sb.toString());
    }
    return grams;
  }
  public static String get3GramHashes(String str) {
    StringBuilder grams = new StringBuilder();
    String[] terms = str.split(" ");
    for(String t: terms) {
      if(t.trim().length()>0) {
        t = "$"+t.trim()+"$";
        List<String> hashes = getCharNGrams(t, 3);
        for(String g: hashes) {
          grams.append(g);
          grams.append(" ");
        }
      }
    }
    return grams.toString();
  }
	public String parse(String str) {
		try {
      str = this.terrier.pipelineText(str);

			return get3GramHashes(str).trim();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
	public String clean(String str) {
		return str.replaceAll(" +", " ").replaceAll(" ", "_");
	}
  public static void main(String[] args) {
    Tokeniser tokeniser = new WordHashTokeniser();
    List<PreProcessTerm> pipeline = new LinkedList<PreProcessTerm>();
//		pipeline.add(PreProcessTerm.SW_REMOVAL);
//		pipeline.add(PreProcessTerm.STEM);
    
    tokeniser.setup(Language.EN, "/home/parth/workspace/terrier-3.5/", pipeline);

    String test = "# kill all men";
    System.out.printf("Input: %s\n", test);
    System.out.printf("Parsed Text: %s\n", tokeniser.parse(test));
    System.out.printf("Cleaned Text: %s\n", tokeniser.clean(tokeniser.parse(test)));

  }

}
