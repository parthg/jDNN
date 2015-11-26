package jair;

import java.util.List;

import java.io.PrintWriter;
import java.io.IOException;

import es.upv.nlel.wrapper.TerrierWrapper;
import es.upv.nlel.corpus.FIRE;
import es.upv.nlel.corpus.Topic;
import es.upv.nlel.corpus.Topic.Tag;

public class MTFireTopics {
  TerrierWrapper terrier;
  List<Topic> topics;

  public static void main(String[] args) throws IOException{
    MTFireTopics mt = new MTFireTopics();
    String queryFile = "/home/parth/workspace/data/fire/topics/en.topics.2011-12.txt";
    String lang = "en";
    boolean stopword_removal = true;
    boolean stem = true;
		String path_to_terrier = "/home/parth/workspace/terrier-3.5/";
		mt.terrier = new TerrierWrapper("/home/parth/workspace/terrier-3.5/");
    String outFile = "/home/parth/Dropbox/mt-autoencoder/sub-corpus/en-topics-to-mt-2011-12.txt";
		
		// Setting up the Pipeline
    mt.terrier.setLanguage(lang);
		if(stopword_removal)
			mt.terrier.setStopwordRemoval(lang);
		if(stem)
			mt.terrier.setStemmer(lang);
	
    // load queries
    mt.topics = FIRE.parseTopicFile(queryFile);

    PrintWriter p = new PrintWriter(outFile, "UTF-8");
    for(Topic t: mt.topics) {
      int id = t.getID();
      String title = mt.terrier.pipelineText(t.get(Tag.TITLE));
      String desc = mt.terrier.pipelineText(t.get(Tag.DESC));
      String narr = mt.terrier.pipelineText(t.get(Tag.NARR));
      p.println(id+"\n"+title.trim()+"\n"+desc.trim()+"\n"+narr.trim());
    }
    p.close();
  }

}
