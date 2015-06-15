package data;

import java.util.regex.Pattern;
import java.util.Set;
import java.util.HashSet;
import java.util.Arrays;
import es.upv.nlel.utils.Language;

public class CleanData {
  public static Set<Integer> ES_CHARS = new HashSet<Integer>(Arrays.asList(0xC1,0xE1, 0xC9, 0xE9, 0xCD, 0xED, 0xD1, 0xF1, 0xD3, 0xF3, 0xDA, 0xFA, 0xDC, 0xFC, 0xAB, 0xBB, 0xBF, 0xA1, 0x80, 0x20A7));
  public static Set<Integer> DE_CHARS = new HashSet<Integer>(Arrays.asList(0xc4, 0xe4, 0xc9, 0xe9, 0xd6, 0xf6, 0xdc, 0xfc, 0xdf, 0xab, 0xbb, 0x84, 0x93, 0x94, 0xb0, 0x80, 0xa3));
	
	public static String parse(String str, Language lang) {
    switch(lang) {
      case EN:
        return parseEnglish(str);
      case HI:
        return parseHindi(str);
      case ES:
        return parseSpanish(str);
      case DE:
        return parseGerman(str);
      default:
        System.err.printf("Option not supported yet! Please set the proper support for language '%s' in CleanData\n", lang.getCode());
        System.exit(0);
        return null;

    }
/*		if(lang == Language.EN)
			return parseEnglish(str);
		else if(lang == Language.HI)
			return parseHindi(str);
		else 
			return null;*/
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
	public static String parseSpanish(String inString) {
		StringBuffer sbf = new StringBuffer();
		char[] str = inString.toCharArray();
		boolean segmented = false;
		for(int i=0; i<str.length; i++) {
			int value = inString.codePointAt(i);
      // If an English character
			if(Pattern.matches("[a-zA-Z]", String.valueOf(str[i]))) {
				sbf.append(Character.toLowerCase(str[i]));
				segmented = false;
			}
			// If a spanish character append
			else if( (ES_CHARS.contains(value) ) 
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
			else if(value>=0x0030 && value<=0x0039) {
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
	public static String parseGerman(String inString) {
		StringBuffer sbf = new StringBuffer();
		char[] str = inString.toCharArray();
		boolean segmented = false;
		for(int i=0; i<str.length; i++) {
			int value = inString.codePointAt(i);
      // If an English character
			if(Pattern.matches("[a-zA-Z]", String.valueOf(str[i]))) {
				sbf.append(Character.toLowerCase(str[i]));
				segmented = false;
			}
			// If a spanish character append
			else if( (DE_CHARS.contains(value) ) 
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
			else if(value>=0x0030 && value<=0x0039) {
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
		System.out.println(parseSpanish("España"));
	}
}
