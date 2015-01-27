package util;

import java.io.File;

public class StringUtils {
  public static String PATH_SEPARATOR = File.separator;

  /** abc/xyz/1.txt -> 1.txt
   */
  public static String fileName(String fullPath) {
    return fullPath.substring(fullPath.lastIndexOf(PATH_SEPARATOR)+1);
  }

  public static String removeExt(String fName) {
    return fName.substring(0, fName.lastIndexOf("."));
  }
}
