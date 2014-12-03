package data;

import java.util.List;
import java.util.Map;

import es.upv.nlel.corpus.NEWSDocType;
import es.upv.nlel.utils.Language;

/** Channel Interface is responsible for input channel of data. Input channel can be a directory for which all the files with a particular extension or a per-line sentence file.
 * To use it, fist make an instance of it with the implementing class (DocCollection or SentFile), setup and setParser.
 *
 * @author Parth Gupta
 */
public interface Channel {
	public void setParser(NEWSDocType type);
	public void setup(TokenType tokenType, Language _lang, String path_to_terrier, List<PreProcessTerm> pipeline);
  /** It would parse the input data and create the lexicon with frequency info. */
	public Map<String, Integer> getTokensFreq();
  /** It would setup an index of the tokens from an external source. Note that Channel doesn't provide to create an index on its own.*/
	public void setTokensIndex(Map<String, Integer> _tokens);
  /** Parses the channel again and creates matrix based on the porivded index. The starting index is from 0.*/
	public Map<Integer, Map<Integer, Integer>> getMatrix();
	public Map<Integer, String> getDataIndex();
//	public String[] getRandomSentences(int n);
}
