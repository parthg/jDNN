package data;

public class TokeniserFactory {
	public static Tokeniser getTokeniser(TokenType type) {
		switch(type) {
		case WORD:
			return new WordTokeniser();
    case WORD_HASH:
      return new WordHashTokeniser();
		default:
			return null;
		}
	}
}
