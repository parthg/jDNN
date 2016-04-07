package util;

import java.util.List;
import java.util.ArrayList;

import java.io.File;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.IOException;

public class FileUtils {
  public static List<String> getLines(File f) throws IOException {
    List<String> lines = new ArrayList<String>();
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"));
    String line = "";
    while((line = br.readLine())!=null) {
      lines.add(line.trim());
    }
    br.close();
    return lines;
  }
}
