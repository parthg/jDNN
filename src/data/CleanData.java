package data;

import java.util.regex.Pattern;

import es.upv.nlel.utils.Language;

public class CleanData {
	
	public static String parse(String str, Language lang) {
		if(lang == Language.EN)
			return parseEnglish(str);
		else if(lang == Language.HI)
			return parseHindi(str);
		else 
			return null;
	}
	public static String parseEnglish(String inString) {
		StringBuffer sbf = new StringBuffer();
		char[] str = inString.toCharArray();
		boolean segmented = false;
		for(int i=0; i<str.length; i++) {
			int value = inString.codePointAt(i);
			if(Pattern.matches("[a-zA-Z]", String.valueOf(str[i]))) {
				sbf.append(Character.toLowerCase(str[i]));
				segmented = false;
			}
			else if(str[i] == ' ') {
				sbf.append(str[i]);
				segmented = false;
			}
			else if(value>=0x0030 && value<=0x0039) {
				if(!segmented) {
					sbf.append(' ');
					segmented = true;
				}
				sbf.append(str[i]);
			}
			else if (Pattern.matches("\\p{Punct}", String.valueOf(str[i]))) {	
				if(i>0 && i<str.length-2 && inString.codePointAt(i-1)>=0x0030 && inString.codePointAt(i-1)<=0x0039 && inString.codePointAt(i+1)>=0x0030 && inString.codePointAt(i+1)<=0x0039) {
					continue;
				}
				else {
					sbf.append(' ');
					segmented = false;
				}
		}
		}
		return sbf.toString().trim().replaceAll(" +", " ").replaceAll("[\\u0030-\\u0039]+", "N").replaceAll(" ", "_");
	}
	public static String parseHindi(String inString) {
		StringBuffer sbf = new StringBuffer();
		char[] str = inString.toCharArray();
		boolean segmented = false;
		for(int i=0; i<str.length; i++) {
			int value = inString.codePointAt(i);
			// If a hindi character append
			if( (value >= 0x0902 && value <= 0x0965) 
					// || (value >= 0x0970 && value <= 0x097f)
					) {
				sbf.append(str[i]);
				segmented = false;
			}
			// append if whitespace
			else if(str[i] == ' ') {
				sbf.append(str[i]);
				segmented = false;
			}
			// if a number and not segmented then add a whitespace before it
			else if((value>=0x0966 && value <= 0x096f) || (value>=0x0030 && value<=0x0039)) {
				if(!segmented) {
					sbf.append(' ');
					segmented = true;
				}
				sbf.append(str[i]);
			}
			// check if the decimal point is defining a number
			else if (Pattern.matches("\\p{Punct}", String.valueOf(str[i]))) {	
					if(i>0 && i<str.length-2 && inString.codePointAt(i-1)>=0x0030 && inString.codePointAt(i-1)<=0x0039 && inString.codePointAt(i+1)>=0x0030 && inString.codePointAt(i+1)<=0x0039) {
						continue;
					}
					else {
						sbf.append(' ');
						segmented = false;
					}
			}
		}
		return sbf.toString().trim().replaceAll("[\\u0966-\\u096f\\u0030-\\u0039]+", " N ").replaceAll(" +", " ").replaceAll(" ", "_");
	}
	public static void main(String[] args) {
		System.out.println(parseHindi("ईरान में शिया सफवी वंश(१५०१-१७२२और भारत में दिल्ली सुल्तानों (१२०६-१५२७) और बाद में मुग़ल साम्राज्१५२६-   हुकूमत हो गयी।"));
	}
}