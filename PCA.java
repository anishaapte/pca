import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;

import Jama.Matrix;

public class PCA {
    public static final int K_LINES = 25;
    public static final int COL_NUM = 784;
    public static final int N_IMAGES = 100;
    
    private double[][] pca;
    private String[][] pcaStr;
    private double[][] nImages;
    private double[][] unitVectors;
    private double[][] projLengths;
    private String[][] projLengthsStr;
    private double[][] reconstruc;
    private String[][] reconstrucStr;
    private double[][] pcaFeat;
    private String[][] pcaFeatStr;
    private double[][] reconImage;
    private String[][] reconImageStr;    
    private double[][] trainingSetFeat;
    private String[][] trainingSetFeatStr;
    private double[][] trainingSetImage;
    private String[][] trainingSetImageStr;
    private double[][] pcaRecon;
    private String[][] pcaReconStr;

    
    public static void main(String[] args) {
        PCA p = new PCA();
        p.processText("/Users/anishaapte/Desktop/AI540/pca.csv", 
                "/Users/anishaapte/Desktop/AI540/test.csv");
        p.kUnitVectors();
        p.projLengths();
        p.pcaFeat();
        p.useTrainingSet("/Users/anishaapte/Desktop/AI540/mnist_train.csv");
        p.reconstruction();
        p.printAnswers();

    }
    public PCA() {
        
    }
    
    public void printAnswers() {
        System.out.println("Q1===========");
        System.out.println();
        for(int i = 0; i < unitVectors.length; i++) {
            printArr(Arrays.toString(unitVectors[i]));
        }
        System.out.println();
        
        System.out.println("Q2===========");
        System.out.println();
        for (int i = 0; i < projLengthsStr.length; i++) {
            printArr(Arrays.toString(projLengthsStr[i]));
        }
        System.out.println();
        
        System.out.println("Q3===========");
        System.out.println();
        for (int i = 0; i < reconstrucStr.length; i++) {
            printArr(Arrays.toString(reconstrucStr[i]));
        }
        System.out.println();
        
        System.out.println("Q4===========");
        System.out.println();
        for (int i = 0; i < pcaStr.length; i++) {
            printArr(Arrays.toString(pcaStr[i]));
        }
        
        System.out.println();
        
        System.out.println("Q5===========");
        System.out.println();
        for (int i = 0; i < pcaFeatStr.length; i++) {
            printArr(Arrays.toString(pcaFeatStr[i]));
        }
        
        System.out.println();
        
        System.out.println("Q6===========");
        System.out.println();
        for (int i = 0; i < reconImageStr.length; i++) {
            printArr(Arrays.toString(reconImageStr[i]));
        }
        System.out.println();
        
        System.out.println("Q7===========");
        System.out.println();
        for (int i = 0; i < trainingSetFeatStr.length; i++) {
            printArr(Arrays.toString(trainingSetFeatStr[i]));
        }
        System.out.println();
        
        System.out.println("Q8===========");
        System.out.println();
        for (int i = 0; i < trainingSetImageStr.length; i++) {
            printArr(Arrays.toString(trainingSetImageStr[i]));
        }
        System.out.println();
        
        System.out.println("Q9===========");
        System.out.println();
        for (int i = 0; i < pcaReconStr.length; i++) {
            printArr(Arrays.toString(pcaReconStr[i]));
        }


    }
    
    public void printArr(String s) {
        System.out.println(s.substring(1, s.length() - 1));
    }
    
    public void processText(String filename, String filename2) {
        pca = new double[K_LINES][COL_NUM];
        pcaStr = new String[K_LINES][COL_NUM];
        nImages = new double[N_IMAGES][COL_NUM];
        try {
            FileReader f = new FileReader(filename);
            FileReader fr = new FileReader(filename2);
            BufferedReader br = new BufferedReader(f);
            BufferedReader b = new BufferedReader(fr);
  
            for (int i = 0; i < K_LINES; i++) {
                String current = br.readLine();
                String[] tokens = current.split(",");
                for (int j = 0; j < COL_NUM; j++) {
                    pca[i][j] = Double.parseDouble(tokens[j]);
                    pca[i][j] = ((double) (Math.round(pca[i][j] * 10000.0))) / 10000.0;
                    pcaStr[i][j] = String.format("%.4f", pca[i][j]);
                }
            }
            for (int i = 0; i < N_IMAGES; i++) {
                String current = b.readLine();
                String[] tokens = current.split(",");
                for (int j = 0; j < COL_NUM; j++) {
                    nImages[i][j] = Double.parseDouble(tokens[j]);
                }
            }
 
            f.close();                    
            br.close();
            
              
        }
        catch(Throwable e) {
            e.printStackTrace();
        }
    }
    
    public void kUnitVectors() {
        unitVectors = new double[K_LINES][COL_NUM];
        for (int i = 0; i < unitVectors.length; i++) {
            for (int j = 0; j < unitVectors[i].length; j++) {
                if (i == j) {
                    unitVectors[i][j] = 1;
                }
                else {
                    unitVectors[i][j] = 0;
                }              
            }
        }        
    }
    
    public void projLengths() {
        projLengthsStr = new String[N_IMAGES][K_LINES];
        reconstrucStr = new String[N_IMAGES][COL_NUM];

        Matrix axes = new Matrix(unitVectors);
        Matrix nImage = new Matrix(nImages);    
        axes = axes.transpose();
        Matrix lengths = nImage.times(axes);
        projLengths = lengths.getArrayCopy();
        
        for (int i = 0; i < projLengths.length; i++) {
            for (int j = 0; j < projLengths[i].length; j++) {
                projLengths[i][j] = ((double) (Math.round(projLengths[i][j] * 10000.0))) / 10000.0;
                projLengthsStr[i][j] = String.format("%.4f", projLengths[i][j]);
            }
        }
        axes = axes.transpose();
        Matrix recon = lengths.times(axes);
        reconstruc = recon.getArrayCopy();
        
        for (int i = 0; i < reconstruc.length; i++) {
            for (int j = 0; j < reconstruc[i].length; j++) {
                reconstruc[i][j] = ((double) (Math.round(reconstruc[i][j] * 10000.0))) / 10000.0;
                reconstrucStr[i][j] = String.format("%.4f", reconstruc[i][j]);
            }
            
        }
        
        
    }
    public void pcaFeat() {
        pcaFeatStr = new String[N_IMAGES][K_LINES];
        reconImageStr = new String[N_IMAGES][COL_NUM];
        Matrix pcam = new Matrix(pca);
        Matrix images = new Matrix(nImages);
        pcam = pcam.transpose();
        Matrix feat = images.times(pcam);
        pcaFeat = feat.getArrayCopy();
        
        for (int i = 0; i < pcaFeat.length; i++) {
            for (int j = 0; j < pcaFeat[i].length; j++) {
                pcaFeat[i][j] = ((double) (Math.round(pcaFeat[i][j] * 10000.0))) / 10000.0;
                pcaFeatStr[i][j] = String.format("%.4f", pcaFeat[i][j]);
            }
            
        }
        pcam = pcam.transpose();
        Matrix temp = feat.times(pcam);
        reconImage = temp.getArrayCopy();
        
        for (int i = 0; i < reconImage.length; i++) {
            for (int j = 0; j < reconImage[i].length; j++) {
                reconImage[i][j] = ((double) (Math.round(reconImage[i][j] * 10000.0))) / 10000.0;
                reconImageStr[i][j] = String.format("%.4f", reconImage[i][j]);
            }
            
        }
        
        
        
    }
    public void useTrainingSet(String filename) {
        
        trainingSetFeat = new double[N_IMAGES][];
        trainingSetFeatStr = new String[N_IMAGES][K_LINES];
        trainingSetImage = new double[N_IMAGES][];
        trainingSetImageStr = new String[N_IMAGES][COL_NUM];
        double[] distances = new double[N_IMAGES];
        for (int i=0; i < distances.length; i++) {
            distances[i] = Double.MAX_VALUE;
        }

        
        int training_count = 10000;
        try {
            
            FileReader f = new FileReader(filename);
            BufferedReader br = new BufferedReader(f);
            Matrix pcam = new Matrix(pca);
            pcam = pcam.transpose();
            
            for (int i = 0; i < training_count; i++) {
                
                // get the next image from the training set
                String current = br.readLine();
                String[] tokens = current.split(",");
                double [][] trainImage = new double[1][COL_NUM];
                for (int j = 0; j < COL_NUM; j++) {
                    trainImage[0][j] = Double.parseDouble(tokens[j]);
                    trainImage[0][j] = ((double) (Math.round(trainImage[0][j] * 10000.0))) / 10000.0;
                }
                
                // turn the image into PCA features
                Matrix imgM = new Matrix (trainImage);
                imgM = imgM.times(pcam);
                double[] iFeat = imgM.getArrayCopy()[0];
                for (int j = 0; j < K_LINES; j++) {
                    iFeat[j] = ((double) (Math.round(iFeat[j] * 10000.0))) / 10000.0;
                }
                
                // go through each of 100 pca feature.  Find the distance.  
                // If smaller then before then use this image as the closest to the pca feature
                for (int j = 0; j < pcaFeat.length; j++) {
                    double iDist = euclideanDistance (iFeat, pcaFeat[j]);
                    if (iDist < distances[j]) {
                        distances[j] = iDist;
                        trainingSetFeat[j] = iFeat;
                        trainingSetImage[j] = trainImage[0];
                    }
                }
                
            }
 
            f.close();                    
            br.close();
            
            
            for (int i = 0; i < trainingSetFeat.length; i++) {
                for (int j = 0; j < trainingSetFeat[i].length; j++) {
                    trainingSetFeatStr[i][j] = String.format("%.4f", trainingSetFeat[i][j]);
                }
            }
            for (int i = 0; i < trainingSetImage.length; i++) {
                for (int j = 0; j < trainingSetImage[i].length; j++) {
                    trainingSetImageStr[i][j] = String.format("%.4f", trainingSetImage[i][j]);
                }
            }
            
            
        }
        catch(Throwable e) {
            e.printStackTrace();
            System.exit(0);
        }

    }
    
    public double euclideanDistance (double [] p1, double [] p2) {
        double returnVal = 0.0;
        for(int i=0; i< p1.length; i++) {
            returnVal += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        returnVal = Math.sqrt(returnVal);
        return returnVal;
    }
    
    public void reconstruction() {
        pcaReconStr = new String[N_IMAGES][COL_NUM];
        pcaRecon = new double[N_IMAGES][COL_NUM];
        Matrix p = new Matrix(pca);
        Matrix t = new Matrix(trainingSetFeat);
        Matrix recon = t.times(p);
        pcaRecon = recon.getArrayCopy();
        
        for (int i = 0; i < pcaRecon.length; i++) {
            for (int j = 0; j < pcaRecon[i].length; j++) {
                pcaRecon[i][j] = ((double) (Math.round(pcaRecon[i][j] * 10000.0))) / 10000.0;
                pcaReconStr[i][j] = String.format("%.4f", pcaRecon[i][j]);
            }
        } 
        
    }

}


