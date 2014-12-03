package data;

import java.io.IOException;
import java.util.List;

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
}
