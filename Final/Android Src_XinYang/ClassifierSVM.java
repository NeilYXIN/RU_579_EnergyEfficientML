package com.neilyxin.www.vibauth;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.TextView;
import umich.cse.yctung.androidlibsvm.LibSVM;

import java.io.*;
import java.math.BigDecimal;

class ClassifierSVM {
    private LibSVM svm;
    private FileWriter fw;
    private BufferedWriter bw;
    private String appFolderPath;
    private String dataPredictName = "SVM_predict.txt";
    private String SVMDataPredictPath;
    private String SVMModelPath;
    private String SVMModelName;
    private String SVMOutputPath;
    private String SVMOutputName;
    private String dataTrainPath;
    private String modelPath;

    private static Handler mainHandler;
    private MySignalProcessing mySignalProcessing;
    private final String TAG = this.getClass().getSimpleName();
    boolean useMFCC;
    boolean useNormalization;

    ClassifierSVM(String appFolderPath, MySignalProcessing mySignalProcessing, boolean useMFCC, boolean useNormalization){
        this.appFolderPath = appFolderPath;
        // Don't omit space for it will appear as a command!
        SVMDataPredictPath = appFolderPath + "SVM_predict.txt ";
        SVMModelPath = appFolderPath + "SVM_model ";
        SVMModelName = appFolderPath + "SVM_model";
        SVMOutputPath = appFolderPath + "SVM_output ";
        SVMOutputName = appFolderPath + "SVM_output";
        dataTrainPath = appFolderPath + "SVM_train.txt ";
        modelPath = appFolderPath + "SVM_model ";
        svm = new LibSVM();
        mainHandler = new Handler(Looper.getMainLooper());
        this.mySignalProcessing = mySignalProcessing;
        this.useMFCC = useMFCC;
        this.useNormalization = useNormalization;
    }

    int predict(Number[] seriesDataNumbers, float[] MFCCResult, final TextView txt_main) {
        int predictionLabel=-1;
        File predicting = new File(appFolderPath + dataPredictName);
        if(!predicting.exists()){
            try {
                predicting.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            fw = new FileWriter(predicting,false);
            bw = new BufferedWriter(fw);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (seriesDataNumbers[0].doubleValue() != 0.0) {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.append(1);

            float[] finalFFT = new float[seriesDataNumbers.length];
            if (useNormalization) {
                finalFFT = mySignalProcessing.normalize(seriesDataNumbers);
            } else {
                for(int i = 0; i < seriesDataNumbers.length; i++) {
                    finalFFT[i] = seriesDataNumbers[i].floatValue();
                }

            }
            for (int i = 0; i < seriesDataNumbers.length; i++) {
                stringBuilder.append(" " + (i+1) + ":" + String.format("%.6f",finalFFT[i]));
            }
            if (useMFCC) {
                float[] finalMFCC = MFCCResult;
                if (useNormalization) {
                    finalMFCC = mySignalProcessing.normalize(MFCCResult);
                }

                for (int i = seriesDataNumbers.length; i < seriesDataNumbers.length + MFCCResult.length; i++) {
                    stringBuilder.append(" " + (i+1) + ":" + String.format("%.6f",finalMFCC[i-seriesDataNumbers.length]));
                }

            }

            stringBuilder.append("\n");
            try {
                bw.write(stringBuilder.toString());
                bw.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        svm.predict(SVMDataPredictPath + SVMModelPath + SVMOutputPath);

        File output = new File(SVMOutputName);
        if (output.exists()) {
            BufferedReader reader = null;
            try {
                reader = new BufferedReader(new FileReader(output));
                String tempString = null;
                int line = 1;
                // 一次读入一行，直到读入null为文件结束
                if ((tempString = reader.readLine())!=null) {
                    final String finalTempString = tempString;
                    predictionLabel = Integer.parseInt(finalTempString);
                }
                reader.close();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                if (reader != null) {
                    try {
                        reader.close();
                    } catch (IOException e1) {
                        Log.e(TAG, "Close BufferReader Failed!!");
                    }
                }
            }
        }
        return predictionLabel;
    }

    void writeSVMTrainingData(Number[] FFTresult, float[] MFCCResult, BufferedWriter mBufferWriter, int label) {
        Number[] seriesDataNumbers = mySignalProcessing.window(FFTresult);

        float[] finalFFT = new float[seriesDataNumbers.length];

        if (seriesDataNumbers[0].doubleValue() != 0.0) {
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.append(label);

            if (useNormalization) {
                finalFFT = mySignalProcessing.normalize(FFTresult);
            } else {
                for (int i = 0; i < seriesDataNumbers.length; i++) {
                    finalFFT[i] = seriesDataNumbers[i].floatValue();
                }
            }
            for (int i = 0; i < seriesDataNumbers.length; i++) {
                stringBuilder.append(" " + (i+1) + ":" + new BigDecimal(String.format("%.6f",finalFFT[i])));
            }

            if (useMFCC) {
                float[] finalMFCC = MFCCResult;
                if (useNormalization) {
                    finalMFCC = mySignalProcessing.normalize(MFCCResult);

                }
                for (int i = seriesDataNumbers.length; i < seriesDataNumbers.length + MFCCResult.length; i++) {
                    stringBuilder.append(" " + (i+1) + ":" + String.format("%.6f",finalMFCC[i-seriesDataNumbers.length]));
                }
            }


            stringBuilder.append("\n");
            try {
                mBufferWriter.write(stringBuilder.toString());
                mBufferWriter.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    void train(String svmTrainOptions) {
        svm.train(svmTrainOptions + dataTrainPath + modelPath);
    }

    String getSVMModelName() {
        return this.SVMModelName;
    }

}