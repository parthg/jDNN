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
    Model enModel = new AddModel();
   
    Language lang = Language.EN;
    String prefix = args[0];
    boolean test = true;
    boolean randomize = true;
    boolean trainContinue=true;
    int lastIter = 10;
    boolean fillDict = true;
//    String modelFile = "obj/tanh-cl-w-0.1-b-100-h-128/model_iter78.txt";
    String modelFile = "obj/tanh-cl-w-0.1-40k-b-100-h-128/model_iter10.txt";
//    String dictFile = "obj/"+prefix+"/dict.txt";
    String useDict = "data/fire/en/dict-parallel.txt";
    double initialCGStepSize = 0.01, finalCGStepSize = 0.01;


    String modelDir = "obj/"+prefix+"/";
    if(!new File(modelDir).exists())
      new File(modelDir).mkdirs();

    String enFile = "data/fire/joint-full/DNN-subparallel-en.dat";
    String hiFile = "data/fire/joint-full/DNN-subparallel-projected-hi.dat";

    String enTestFile = "data/fire/joint-full/DNN-subparallel-en-test-part.dat";
    String hiTestFile = "data/fire/joint-full/DNN-subparallel-projected-hi-test-part.dat";

    Corpus enCorp = loadCorpus(enFile);
    DMatrix hiPos = loadMatrix(128, hiFile);

    Corpus enCorpTest = loadCorpus(enTestFile);
    DMatrix hiTestPos = loadMatrix(128, hiTestFile);
    
    System.out.printf("Total Train Sentences = %d \n", enCorp.getSize());

    System.out.printf("Total Test Sentences = %d \n", enCorpTest.getSize()); 

    // ********** DICTIONARY ************* //
    Dictionary enDict = new Dictionary();
    if(trainContinue || useDict.length()>0) {
      enDict.load(useDict);
      fillDict = false;
    }
    
    if(trainContinue) {
      enModel.load(modelFile, enDict);
      System.out.printf("Model loaded to continue training from iteration = %d\n", (lastIter+1));
    }
    else {
      enModel.setDict(enDict);
      enDict.save(modelDir+"dict.txt");
//      Layer l = new TanhLayer(300);
//      enModel.addHiddenLayer(l);

      Layer l2 = new TanhLayer(128);
      enModel.addHiddenLayer(l2);
  
      enModel.init(0.1, 0.0);
      enModel.printArchitecture();
    }
    int[] randArray = new int[enCorp.getSize()];
    for(int i=0; i<enCorp.getSize(); i++)
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
    for(int i=0; i<enCorpTest.getSize(); i++) {
      Sentence s = enCorpTest.get(i);
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

    DMatrix posMatTest = DMath.createMatrix(posIndexTest.size(), hiTestPos.columns());
    DMatrix negMatTest = DMath.createMatrix(negIndexTest.size(), hiTestPos.columns());
    for(int i=0; i< posIndexTest.size(); i++) {
      posMatTest.fillRow(i, hiTestPos.getRow(posIndexTest.get(i)));
      negMatTest.fillRow(i, hiTestPos.getRow(negIndexTest.get(i)));
    }

//    System.out.printf("Finally, train instances = %d and test instances = %d\n", instances.size(), test_instances.size());

    try(Batch testBatch = new Batch(test_instances, 1, enModel.dict(), posMatTest, negMatTest)) {
      
      int batchsize = 100;
      int iterations = 400;

      for(int iter = lastIter+1; iter<=iterations; iter++) {
        int batchNum = 1;

        if(randomize)
          RandomUtils.suffleArray(randArray);

        List<Datum> instances = new ArrayList<Datum>();
        List<Integer> posIndex = new ArrayList<Integer>();
        count = 0;
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
          negIndex.add(negArray[i]);
    

        System.out.printf("\n\nIteration = %d", iter);
        for(int i=0; i<instances.size(); i+=batchsize) {
          int innerbatchsize = batchsize;
          System.out.printf("\nBatch = %d ", batchNum);
          int left = instances.size()-i;
          if(left<batchsize)
            innerbatchsize=left;
          
          List<Datum> batch = new ArrayList<Datum>();
          DMatrix batchPos = DMath.createMatrix(innerbatchsize, hiPos.columns());
          DMatrix batchNeg = DMath.createMatrix(innerbatchsize, hiPos.columns());

          for(int j=0; j<innerbatchsize; j++) {
            batch.add(instances.get(i+j));
            batchPos.fillRow(j, hiPos.getRow(posIndex.get(i+j)));
            batchNeg.fillRow(j, hiPos.getRow(negIndex.get(i+j)));
          } 
          try(Batch matBatch = new Batch(batch, 1, enModel.dict(), batchPos, batchNeg);) {
            matBatch.copyHtoD();
            GradientCalc trainer = new CLNoiseCosineGradientCalcDeep(matBatch);
            trainer.setModel(enModel);
            // MAXIMISER
            Optimizer optimizer = null;
            if(iter>5)
              optimizer = new ConjugateGradient(trainer, finalCGStepSize);
            else
              optimizer = new ConjugateGradient(trainer, initialCGStepSize);
            optimizer.optimize(3);
            double[] learntParams = new double[enModel.getThetaSize()];
            trainer.getParameters(learntParams);
            enModel.setParameters(learntParams);
            batchNum++;
/*            GradientCheck gCheck = new GradientCheck(new CLNoiseCosineGradientCalcDeep(matBatch));
            gCheck.optimise(enModel);*/
            matBatch.close();
            if(batchNum%100==0) {
              trainer.testStats(testBatch);
              System.out.printf("\nAfter Batch %d Test Cost = %.6f and Test MRR = %.6f\n\n", batchNum, trainer.testLoss(), trainer.testMRR());
            }
            if(SimpleCuBlas.cudaCount > 0)
              System.out.printf("At end of batch cudaCount = %d\n", SimpleCuBlas.cudaCount);

            SimpleCuBlas.reset();
          } finally {
            enModel.clearDevice();
          }
           
        }
        if(test) {
          GradientCalc trainer = new CLNoiseCosineGradientCalcDeep(null);
          trainer.setModel(enModel);
          trainer.testStats(testBatch);
          System.out.printf("After Iteration %d Cost = %.6f and MRR = %.6f\n\n", (iter), trainer.testLoss(), trainer.testMRR());
        }
        enModel.clearDevice();
        enModel.save(modelDir+"model_iter"+iter+".txt");
        SimpleCuBlas.reset();
      } // for iterations closed

      testBatch.close();
      SimpleCuBlas.reset();
    }// try test closed
  } // main closed
}
