package models;

import models.Model;

import models.AddModel;
import data.PreProcessTerm;
import data.Channel;
import data.SentFile;
import data.TokenType;

import nn.Layer;
import nn.LogisticLayer;
import nn.TanhLayer;

import math.DMath;
import math.DMatrix;

import es.upv.nlel.utils.Language;

import common.Sentence;
import common.Corpus;
import common.Dictionary;
import common.Datum;
import common.Batch;

import optim.GradientCheck;
import optim.GradientCalc;
//import optim.BasicGradientCalc;
import optim.NoiseGradientCalc;
import optim.NoiseCosineGradientCalc;
import optim.CLNoiseCosineGradientCalc;
import optim.CLNoiseCosineGradientCalcDeep;
import optim.NoiseCosineGradientCalcDeep;


import math.jcublas.SimpleCuBlas;

import cc.mallet.optimize.ConjugateGradient;
import cc.mallet.optimize.Optimizer;

import random.RandomUtils;

import java.util.List;
import java.util.ArrayList;
import java.io.IOException;
import java.io.File;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;


public class CLTrainModel {

  public static Corpus loadCorpus(String file) throws IOException {
    Corpus corp = new Corpus();
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
    String line = "";
    while((line = br.readLine())!=null) {
      Sentence s = new Sentence();
      String[] cols = line.split(" ");
      for(int i=0; i<cols.length; i++)
        s.addWord(Integer.parseInt(cols[i].trim()));
      corp.addSent(s);
    }
    return corp;
  }
  public static DMatrix loadMatrix(int dim, String file) throws IOException {
    DMatrix m = DMath.createMatrix(5000, dim);
    BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
    String line = "";
    int i=0;
    while((line = br.readLine())!=null) {
      if(i>=m.rows())
        m.inflateRows(m.rows()+5000, m.columns());
      String[] cols = line.split(" ");
      for(int j=0; j<cols.length; j++)
        m.put(i, j, Double.parseDouble(cols[j].trim()));
      i++;
    }
    if(m.rows()>i)
      m.truncateRows(i, m.columns());
    return m;

  }

  public static void main(String[] args) throws IOException {
    if(args.length!=1) {
      System.out.printf("Usage: sh run.sh models.CLTrainModel <prefix>\n");
      System.exit(0);
    }
    Model lang1Model = new AddModel();
  
    String corpus = "clef"; // "fire-new" OR "fire-new"
    Language lang1 = Language.DE; // query lang
    Language lang2 = Language.EN; // doc lang
    String prefix = args[0];
    boolean test = true;
    boolean randomize = true;
    boolean trainContinue=true;
    int lastIter = 6;
    boolean fillDict = false;
//    String modelFile = "obj/tanh-cl-w-0.1-b-100-h-128/model_iter78.txt";
    String modelFile = "obj/tanh-clef-de-en-cl-w-0.1-10k-b-100-h-128-bias-1.5-new/model_iter6.txt";
//    String dictFile = "obj/"+prefix+"/dict.txt";
    String useDict = "data/"+corpus+"/"+lang1.getCode()+"/dict-top10000.txt";
    double initialCGStepSize = 0.01, finalCGStepSize = 0.01;


    String modelDir = "obj/"+prefix+"/";
    if(!new File(modelDir).exists())
      new File(modelDir).mkdirs();

    String lang1File = "data/"+corpus+"/joint-de/DNN-subparallel-"+lang1.getCode()+".dat";
    String lang2File = "data/"+corpus+"/joint-de/DNN-subparallel-projected-"+lang2.getCode()+".dat";

    String lang1TestFile = "data/"+corpus+"/joint-de/DNN-subparallel-"+lang1.getCode()+"-test-part.dat";
    String lang2TestFile = "data/"+corpus+"/joint-de/DNN-subparallel-projected-"+lang2.getCode()+"-test-part.dat";

    Corpus lang1Corp = loadCorpus(lang1File);
    DMatrix lang2Pos = loadMatrix(128, lang2File);

    Corpus lang1CorpTest = loadCorpus(lang1TestFile);
    DMatrix lang2TestPos = loadMatrix(128, lang2TestFile);
    
    System.out.printf("Total Train Sentences = %d \n", lang1Corp.getSize());

    System.out.printf("Total Test Sentences = %d \n", lang1CorpTest.getSize()); 

    // ********** DICTIONARY ************* //
    Dictionary lang1Dict = new Dictionary();
    if(trainContinue || useDict.length()>0) {
      lang1Dict.load(useDict);
      fillDict = false;
    }
    
    if(trainContinue) {
      lang1Model.load(modelFile, lang1Dict);
      System.out.printf("Model loaded to continue training from iteration = %d\n", (lastIter+1));
    }
    else {
      lang1Model.setDict(lang1Dict);
      lang1Dict.save(modelDir+"dict.txt");
      Layer l = new TanhLayer(128);
      lang1Model.addHiddenLayer(l);

/*      Layer l2 = new TanhLayer(128);
      lang1Model.addHiddenLayer(l2);*/
  
      lang1Model.init(0.1, -1.5);
      lang1Model.printArchitecture();
    }
    int[] randArray = new int[lang1Corp.getSize()];
    for(int i=0; i<lang1Corp.getSize(); i++)
      randArray[i] = i;
    
/*    if(randomize)
      RandomUtils.suffleArray(randArray);

    List<Datum> instances = new ArrayList<Datum>();
    List<Integer> posIndex = new ArrayList<Integer>();
    int count = 0;
    for(int i=0; i<enCorp.getSize(); i++) {
      Sentence s = enCorp.get(randArray[i]);
      if(s.getSize()>0) {
        Datum d = new Datum(count, s);
        instances.add(d);
        posIndex.add(randArray[i]);
        count++;
      }
    }

    int[] negArray = new int[posIndex.size()];
    for(int i=0; i< posIndex.size(); i++)
      negArray[i] = posIndex.get(i);
    RandomUtils.suffleArray(negArray);
    List<Integer> negIndex = new ArrayList<Integer>();
    for(int i=0; i< negArray.length; i++)
      negIndex.add(negArray[i]);*/


    List<Datum> test_instances = new ArrayList<Datum>();
    List<Integer> posIndexTest = new ArrayList<Integer>();
    int count = 0;
    for(int i=0; i<lang1CorpTest.getSize(); i++) {
      Sentence s = lang1CorpTest.get(i);
      if(s.getSize()>0) {
        Datum d = new Datum(count, s);
        test_instances.add(d);
        posIndexTest.add(i);
        count++;
      }
    }
    
    int[] negArrayTest = new int[posIndexTest.size()];
    for(int i=0; i< posIndexTest.size(); i++)
      negArrayTest[i] = posIndexTest.get(i);
    RandomUtils.suffleArray(negArrayTest);
    List<Integer> negIndexTest = new ArrayList<Integer>();
    for(int i=0; i< negArrayTest.length; i++)
      negIndexTest.add(negArrayTest[i]);

    DMatrix posMatTest = DMath.createMatrix(posIndexTest.size(), lang2TestPos.columns());
    DMatrix negMatTest = DMath.createMatrix(negIndexTest.size(), lang2TestPos.columns());
    for(int i=0; i< posIndexTest.size(); i++) {
      posMatTest.fillRow(i, lang2TestPos.getRow(posIndexTest.get(i)));
      negMatTest.fillRow(i, lang2TestPos.getRow(negIndexTest.get(i)));
    }

//    System.out.printf("Finally, train instances = %d and test instances = %d\n", instances.size(), test_instances.size());

    try(Batch testBatch = new Batch(test_instances, 1, lang1Model.dict(), posMatTest, negMatTest)) {
      
      int batchsize =100;
      int iterations = 100;

      for(int iter = lastIter+1; iter<=iterations; iter++) {
        int batchNum = 1;

        if(randomize)
          RandomUtils.suffleArray(randArray);

        List<Datum> instances = new ArrayList<Datum>();
        List<Integer> posIndex = new ArrayList<Integer>();
        count = 0;
        for(int i=0; i<lang1Corp.getSize(); i++) {
          Sentence s = lang1Corp.get(randArray[i]);
          if(s.getSize()>0) {
            Datum d = new Datum(count, s);
            instances.add(d);
            posIndex.add(randArray[i]);
            count++;
          }
        }

        int[] negArray = new int[posIndex.size()];
        for(int i=0; i< posIndex.size(); i++)
          negArray[i] = posIndex.get(i);
        RandomUtils.suffleArray(negArray);
        List<Integer> negIndex = new ArrayList<Integer>();
        for(int i=0; i< negArray.length; i++)
          negIndex.add(negArray[i]);
    

        System.out.printf("\n\nIteration = %d", iter);
        for(int i=0; i<instances.size(); i+=batchsize) {
          int innerbatchsize = batchsize;
          System.out.printf("\nBatch = %d ", batchNum);
          int left = instances.size()-i;
          if(left<batchsize)
            innerbatchsize=left;
          
          List<Datum> batch = new ArrayList<Datum>();
          DMatrix batchPos = DMath.createMatrix(innerbatchsize, lang2Pos.columns());
          DMatrix batchNeg = DMath.createMatrix(innerbatchsize, lang2Pos.columns());

          for(int j=0; j<innerbatchsize; j++) {
            batch.add(instances.get(i+j));
            batchPos.fillRow(j, lang2Pos.getRow(posIndex.get(i+j)));
            batchNeg.fillRow(j, lang2Pos.getRow(negIndex.get(i+j)));
          } 
          try(Batch matBatch = new Batch(batch, 1, lang1Model.dict(), batchPos, batchNeg);) {
            matBatch.copyHtoD();
            GradientCalc trainer = new CLNoiseCosineGradientCalcDeep(matBatch);
            trainer.setModel(lang1Model);
            // MAXIMISER
            Optimizer optimizer = null;
            if(iter>5)
              optimizer = new ConjugateGradient(trainer, finalCGStepSize);
            else
              optimizer = new ConjugateGradient(trainer, initialCGStepSize);
            optimizer.optimize(3);
            double[] learntParams = new double[lang1Model.getThetaSize()];
            trainer.getParameters(learntParams);
            lang1Model.setParameters(learntParams);
            batchNum++;

//            GradientCheck gCheck = new GradientCheck(new CLNoiseCosineGradientCalcDeep(matBatch));
//            gCheck.optimise(lang1Model);

            matBatch.close();
            if(batchNum%100==0) {
              trainer.testStats(testBatch);
              System.out.printf("\nAfter Batch %d Test Cost = %.6f and Test MRR = %.6f\n\n", batchNum, trainer.testLoss(), trainer.testMRR());
            }
            if(SimpleCuBlas.cudaCount > 0)
              System.out.printf("At end of batch cudaCount = %d\n", SimpleCuBlas.cudaCount);

            SimpleCuBlas.reset();
          } finally {
            lang1Model.clearDevice();
          }
           
        }
        if(test) {
          GradientCalc trainer = new CLNoiseCosineGradientCalcDeep(null);
          trainer.setModel(lang1Model);
          trainer.testStats(testBatch);
          System.out.printf("After Iteration %d Cost = %.6f and MRR = %.6f\n\n", (iter), trainer.testLoss(), trainer.testMRR());
        }
        lang1Model.clearDevice();
        lang1Model.save(modelDir+"model_iter"+iter+".txt");
        SimpleCuBlas.reset();
      } // for iterations closed

      testBatch.close();
      SimpleCuBlas.reset();
    }// try test closed
  } // main closed
}
