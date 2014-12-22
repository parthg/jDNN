package retrieval;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;

import org.jblas.DoubleMatrix;

import es.upv.nlel.deep.AutoencoderBLAS;
import es.upv.nlel.math.Matrix;
import es.upv.nlel.preprocess.CharNGramTokeniser;
import es.upv.nlel.utils.Language;
import es.upv.nlel.utils.ValueComparatorAsc;
import es.upv.nlel.utils.ValueComparatorDesc;
import data.CleanData;
import gnu.trove.map.hash.TIntDoubleHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

public class Retrieval {
	AutoencoderBLAS ae;
	Map<String, Integer> features;
	int[] grams;
	Language lang;
  public Retrieval(Language _lang, int[] _grams) {
    this.lang = _lang;
    this.grams = _grams;
  }
	
	double[][] data;
	List<String> docIndex;
	
	public void setGrams(int[] n) {
		this.grams = n;
	}
	public void setLang(Language l) {
		this.lang = l;
	}
	public void loadModel(String modelFile, boolean rsm, String code, int networkSize) throws Exception {
		this.ae = new AutoencoderBLAS(networkSize);
		this.ae.loadWeights(modelFile);
		if(rsm)
			this.ae.setRSM();
		this.ae.setCodeLayer(code);
	}
	public void loadFeatures(String featureFile) throws NumberFormatException, IOException {
		this.features = new HashMap<String, Integer>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(featureFile), "UTF-8"));
		String line = "";
		while((line = br.readLine())!=null) {
			String[] cols = line.split("\t");
			this.features.put(cols[1].trim(), Integer.parseInt(cols[0].trim()));
		}
		br.close();
	}
	public void loadDocIndex(String indexFile) throws IOException {
    this.docIndex =  new ArrayList<String>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(indexFile), "UTF-8"));
		String line = "";
		while((line = br.readLine())!=null) {
			String[] cols = line.split("\t");
			this.docIndex.add(cols[1].trim());
		}
		br.close();
	}
	public void loadProjecteDataObject(String obj) throws IOException, ClassNotFoundException {
		FileInputStream fis = new FileInputStream(obj);
		ObjectInputStream ois = new ObjectInputStream(fis);
		
		System.out.print("[info] Reading the ObjectFile..");
		this.data = (double[][]) ois.readObject();
		
		ois.close();
		fis.close();
		System.out.print("[info] Done. [" + data.length + "x" + data[0].length + "]");
		
	}
	public void writeProjectedDataObject(String obj) throws IOException {
		FileOutputStream fos = new FileOutputStream(obj);
		ObjectOutputStream oos = new ObjectOutputStream(fos);
		
		oos.writeObject(this.data);
		
		oos.close();
		fos.close();
	}
	@SuppressWarnings("unchecked")
	public List<String> getRankList(double[][] inVec, String dist) {
		double[][] sim = Matrix.multiplyTranspose(inVec, this.data);
		System.out.println("Scores calculated with : " + sim[0].length);
		Map<Integer, Double> scores = new HashMap<Integer, Double>();
		for(int i=0; i<sim[0].length; i++)
			scores.put(i, sim[0][i]);
		ValueComparatorDesc bvc =  null;
		ValueComparatorAsc avc = null;
		TreeMap<Integer,Double> sorted_scoremap = null;
		if(dist.equals("cosine")) {
			bvc = new ValueComparatorDesc(scores);
			sorted_scoremap = new TreeMap<Integer,Double>(bvc);
		}
		else if(dist.equals("euclidean")) {
			avc = new ValueComparatorAsc(scores);
			sorted_scoremap = new TreeMap<Integer,Double>(avc);
		}
    sorted_scoremap.putAll(scores);
    int count = 0;
    List<String> rl = new ArrayList<String>();
    for(int j: sorted_scoremap.keySet()) {
    	if(count <10) {
    		rl.add(this.docIndex.get(j));
    		System.out.println(count +"\t" +this.docIndex.get(j) + "\t" + scores.get(j));
    		count++;
    	}
    	else
    		break;
    }
    return rl;
	}
	
	public DoubleMatrix project(TIntObjectHashMap<TIntDoubleHashMap> batch, int l) {
		return this.ae.getNormalisedVector(this.ae.getHiddenActivities(batch, l));
	}
	public void projectData(String inputSparseData, int l) throws IOException {
		TIntObjectHashMap<TIntDoubleHashMap> spMtx = new TIntObjectHashMap<TIntDoubleHashMap>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inputSparseData), "UTF-8"));
		String line = "";
		while((line = br.readLine())!= null) {
			String[] cols = line.split("\t");
			int did = Integer.parseInt(cols[0]);
			int tid = Integer.parseInt(cols[1]);
			double count = Double.parseDouble(cols[2]);
			if(!spMtx.contains(did)) {
				TIntDoubleHashMap inner = new TIntDoubleHashMap();
				inner.put(tid, count);
				spMtx.put(did, inner);
			}
			else {
				TIntDoubleHashMap inner = spMtx.get(did);
				inner.put(tid, count);
				spMtx.put(did, inner);
			}
		}
		br.close();
//		this.data = new double[spMtx.size()][];
		DoubleMatrix dt = new DoubleMatrix();
		TIntObjectHashMap<TIntDoubleHashMap> temp = new TIntObjectHashMap<TIntDoubleHashMap>();
		int k = 0;
		for(int i=0; i<spMtx.size(); i++) {
			if(i%5000==0) {
				System.out.println("Projected : " + dt.rows);
				k=0;
				if(temp.size()>0) {
					System.out.println("Size of batch = " + temp.size());
					DoubleMatrix part = this.project(temp, l); 
					if(dt.length==0) {
						dt = part;
						System.out.println(dt.getRow(0).toString());
					}
					else
						dt = DoubleMatrix.concatVertically(dt, part);
					temp = new TIntObjectHashMap<TIntDoubleHashMap>();
				}
				temp.put(k, spMtx.get(i));
				k++;
//				else {ae.getNumLayers()/2
//					temp.put(k, spMtx.get(i));
//          k++;
//        }
			}
			else {
				temp.put(k, spMtx.get(i));
				k++;
			}
		}
		// the last batch may be less than the fixed-size
		if(temp.size()>0) {
			System.out.println("Size of batch = " + temp.size());
			DoubleMatrix part = this.project(temp, l); 
			//		this.ae.getNormalisedVector(this.ae.getHiddenActivities(temp));
			System.out.println("Size of projected batch = " + part.rows + "x"+ part.columns);
			if(dt.length==0) {
				dt = part;
				System.out.println(dt.getRow(0).toString());
			}
			else
				dt = DoubleMatrix.concatVertically(dt, part);
			temp = new TIntObjectHashMap<TIntDoubleHashMap>();
		}
		
		// finally set it in data
		this.data = dt.toArray2();
		dt = new DoubleMatrix();
	}
	public void loadProjectedData(String matrixFile) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(matrixFile), "UTF-8"));
		String line = "";
		List<double[]> dataList = new ArrayList<double[]>();
		while((line = br.readLine())!=null) {
			String[] cols = line.split(" ");
			double[] f = new double[cols.length];
			for(int i=0; i < cols.length; i++)
				f[i] = Double.parseDouble(cols[i]);
			dataList.add(f);
		}
		this.data = dataList.toArray(new double[dataList.size()][]);
		br.close();
	}
	public String cleanText(String text) {
		return CleanData.parse(text, this.lang);
	}
	public List<String> getFeatureVector(String text) {
    System.out.println("Cleaned Text = " + text);
		List<String> fv = new ArrayList<String>();
		for(int n: this.grams) {
			fv.addAll(CharNGramTokeniser.getSentGrams(text,n));
		}
    System.out.println("Total n-grams = " + fv.size());
		return fv;
	}
	public double[] getLatentVector(String text, int l) {
    System.out.println("Text before: " + text);
		text = this.cleanText(text);
		List<String> fv = this.getFeatureVector(text);
		TIntDoubleHashMap fvTemp = new TIntDoubleHashMap();
		for(String f: fv) {
			if(this.features.containsKey(f)) {
				int id = this.features.get(f);
				if(!fvTemp.contains(id))
					fvTemp.put(id, 1.0);
				else
					fvTemp.put(id, fvTemp.get(id)+1.0);
			}
		}
    System.out.println(fvTemp.size());
//		for(int i:fv.keySet())
//			fvTemp.put(i-1, fv.get(i));	
    TIntObjectHashMap<TIntDoubleHashMap> fvMat = new TIntObjectHashMap<TIntDoubleHashMap>();
    fvMat.put(0, fvTemp);
    double[] vec = this.ae.getNormalisedVector(this.ae.getHiddenActivities(fvMat, l)).toArray2()[0];
		return vec;

	}
	public static void main(String[] args) throws Exception {
		boolean cl = true;
    int[] n = {3};   
    Retrieval ret = new Retrieval(Language.HI, n);
		ret.loadModel("scripts/matlab/output/model/ieee-fire-hi/fine3gramsieeehirsmweights13bin.oct", true, "bin", 8);
		ret.loadFeatures("etc/hi.3grams.txt");
		ret.loadDocIndex("etc/fire/fire-hi-nt-did-map.txt");
//		ret.loadProjectedData("etc/fire/projection/doc-projection.txt");
		String dataObj = "obj/hi-nt-doc.obj";
		if(!new File(dataObj).exists()) {
      ret.projectData("etc/fire/data/whole/train-fire-navbharat-hi.txt", ret.ae.getNumLayers()/2);
      ret.writeProjectedDataObject(dataObj);
		}
		else {
			ret.loadProjecteDataObject(dataObj);
		}
    ret.setGrams(n);
    
    Retrieval ret_en = new Retrieval(Language.EN, n);
    ret_en.loadModel("scripts/matlab/output/model/ieee-fire-en/clfine3gramsieeeenrsmweights13bincosinenoprob.oct", true, "bin", 4);
		ret_en.loadFeatures("etc/en.3grams.txt");
		
		while(true) {
			Scanner in = new Scanner(System.in);
			System.out.print("Input Term: ");
			String input = in.nextLine();
			double[][] vec = new double[1][];
			if(cl) {
				vec[0] = ret_en.getLatentVector(input, ret_en.ae.getNumLayers());
			}
			else
				vec[0] = ret.getLatentVector(input, ret.ae.getNumLayers()/2);
			for(int i=0; i<vec[0].length ; i++)
				System.out.print(vec[0][i]+" ");
			System.out.println();		
			
			List<String> rl = ret.getRankList(vec, "cosine");
			/*for(int i=0; i<10; i++) {
				System.out.println(i+"\t:\t" + rl.get(i));
			}*/
			if(input.equals("exit")) {
				in.close();
				break;
			}
		}
		System.out.println("Exiting..");
	}
}
