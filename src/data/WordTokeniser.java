package data;

import java.io.IOException;
import java.util.List;
import java.util.LinkedList;

import es.upv.nlel.utils.Language;
import es.upv.nlel.wrapper.TerrierWrapper;

public class WordTokeniser implements Tokeniser {
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
	public String parse(String str) {
		try {
			return this.terrier.pipelineText(str);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
	public String clean(String str) {
		return CleanData.parse(str, this.lang);
	}

  public static void main(String[] args) {
    Tokeniser tokeniser = new WordTokeniser();
    List<PreProcessTerm> pipeline = new LinkedList<PreProcessTerm>();
		pipeline.add(PreProcessTerm.SW_REMOVAL);
		pipeline.add(PreProcessTerm.STEM);
    
    tokeniser.setup(Language.ES, "/home/parth/workspace/terrier-3.5/", pipeline);

    String test = "Víctimas de Avalanchas";
    System.out.printf("Input: %s\n", test);
    System.out.printf("Parsed Text: %s\n", tokeniser.parse(test));
    System.out.printf("Cleaned Text: %s\n", tokeniser.clean(tokeniser.parse(test)));

  }
}
