package data;

public class TokeniserFactory {
	public static Tokeniser getTokeniser(TokenType type) {
		switch(type) {
		case WORD:
			return new WordTokeniser();
		default:
			return null;
		}
	}
}