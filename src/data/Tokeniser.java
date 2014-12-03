package data;

import java.util.List;

import es.upv.nlel.utils.Language;
/** The Tokeniser interface allows to generic tokenisation of the text with classes WordTokeniser implementing it. Used in the Channel interface. <br><br>
 * Usage: <br><br>
 *  Tokeniser tokeniser = TokeniserFactory.getTokeniser(tokenType); <br>
 *  tokeniser.setup(_lang, path_to_terrier, pipeline); <br>
 *  tokeniser.parse(txt); <br>
 *  tokeniser.clean(txt); <br>
 *
 * @author Parth Gupta
 */
public interface Tokeniser {
	public void setup(Language _lang, String path_to_terrier, List<PreProcessTerm> pipeline);
	public String parse(String str);
	public String clean(String str);
}
