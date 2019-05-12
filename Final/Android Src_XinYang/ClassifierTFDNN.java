package com.neilyxin.www.vibauth;

import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.os.Environment;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedWriter;
import java.io.IOException;
import java.math.BigDecimal;


class ClassifierTFDNN {
    private static final String TAG = "PredictionTF";
    // Input Dimension
    private static final int IN_COL = 1;
//    private static final int IN_ROW = 256;
    private int IN_ROW;
    private static int OUT_COL; // number of labels
    private static final int OUT_ROW = 1;
    // Name of the Input Node
    private static final String inputName = "input/x_input";
    // Name of the Output Node
    private static final String outputName = "output_layer/output/output";

    String filepath = Environment.getExternalStorageDirectory().getPath();

    String appFolderPath = filepath + "/VibAuth/";
    String modelFileName = "VibAuthModel.pb";

    TensorFlowInferenceInterface inferenceInterface;

    boolean useMFCC;
    boolean useNormalization;

    ClassifierTFDNN(boolean useMFCC, boolean useNormalization) {
        this.useMFCC = useMFCC;
        this.useNormalization = useNormalization;
    }

    ClassifierTFDNN(AssetManager assetManager, String modePath, SharedPreferences sharedPreferences, boolean useMFCC, boolean useNormalization) {
        // Initialize TensorFlowInferenceInterface
        this.useMFCC = useMFCC;
        this.useNormalization = useNormalization;
        int labelsNum = sharedPreferences.getInt("LABELSNUM", -1);
        if (labelsNum == -1) {
            // TODO: Handle Error
            Log.e(TAG, "LABEL SUM ERROR!!");
        } else {
            OUT_COL = labelsNum;
            inferenceInterface = new TensorFlowInferenceInterface(assetManager,modePath);
            Log.d(TAG,"Loading TensorFlow Model File Successfully!");
        }

    }

    float[] predict(Number[] seriesDataNumbers, float[] MFCCResult, MySignalProcessing mySignalProcessing) {
        float[] features;
        if (useMFCC) {
            features = new float[seriesDataNumbers.length + MFCCResult.length];
        } else {
            features = new float[seriesDataNumbers.length];
        }

        float[] finalFFT = new float[seriesDataNumbers.length];
        if (useNormalization) {
            finalFFT = mySignalProcessing.normalize(seriesDataNumbers);
        } else {
            for (int i = 0; i < seriesDataNumbers.length; i++) {
                finalFFT[i] = seriesDataNumbers[i].floatValue();
            }
        }

        for (int i = 0; i < seriesDataNumbers.length; i++) {
            features[i] = Float.parseFloat(String.format("%.6f",finalFFT[i]));
        }


        if (useMFCC) {
            float[] finalMFCC = MFCCResult;
            if (useNormalization) {
                finalMFCC = mySignalProcessing.normalize(MFCCResult);
            }
            for (int i = seriesDataNumbers.length; i < seriesDataNumbers.length+MFCCResult.length; i++) {
                features[i]= Float.parseFloat(String.format("%.6f",finalMFCC[i-seriesDataNumbers.length]));
            }
        }

        // Feed data to the input
        IN_ROW = features.length;

        inferenceInterface.feed(inputName, features, IN_COL, IN_ROW);
        // Run TensorFlow
        String[] outputNames = new String[] {outputName};
        inferenceInterface.run(outputNames);
        // Fetch data from the output
        float[] outputs = new float[OUT_COL*OUT_ROW];
        inferenceInterface.fetch(outputName, outputs);
        int predictionLabel = -1;
        float predictionConfidence = Float.MIN_VALUE;
        for (int i = 0; i < outputs.length; i++) {
            if (outputs[i] > predictionConfidence) {
                predictionConfidence = outputs[i];
                predictionLabel = i;
            }
        }
        return new float[]{predictionLabel, predictionConfidence};
    }

    /**
     * Update the graph when gets new data
     */
    void writeTensorFlowFFTTrainingData(Number[] FFTresult, float[] MFCCResult, BufferedWriter mBufferWriter, int label, MySignalProcessing mySignalProcessing) {
        Number[] seriesDataNumbers = mySignalProcessing.window(FFTresult);
        if (seriesDataNumbers[0].doubleValue() != 0.0) {
            StringBuilder stringBuilder = new StringBuilder();

            float[] finalFFT = new float[seriesDataNumbers.length];
            if (useNormalization) {
                finalFFT = mySignalProcessing.normalize(seriesDataNumbers);
            } else {
                for (int i = 0; i < seriesDataNumbers.length; i++) {
                    finalFFT[i] = seriesDataNumbers[i].floatValue();
                }
            }

//            Log.e("Feature Counter", "FFT: " + normalizedFFT.length);


            for (int i = 0; i < seriesDataNumbers.length; i++) {
                stringBuilder.append(String.format("%.6f", finalFFT[i])).append(",");

            }

            if (useMFCC) {
                // MFCC data
                float[] finalMFCC = MFCCResult;
                if (useNormalization) {
                    finalMFCC = mySignalProcessing.normalize(MFCCResult);
                }
                Log.e("Feature Counter", "MFCC: " + finalMFCC.length);

                for (int i = seriesDataNumbers.length; i < seriesDataNumbers.length + MFCCResult.length; i++) {
                    stringBuilder.append(String.format("%.6f",finalMFCC[i-seriesDataNumbers.length])).append(",");
                }
            }


            stringBuilder.append(label);
            stringBuilder.append("\n");
            try {
                mBufferWriter.write(stringBuilder.toString());
                mBufferWriter.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    /**
     * Update the graph when gets new data
     */
    void writeTensorFlowSignalTrainingData(byte[] signals, BufferedWriter mBufferWriter, int label) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < signals.length; i++) {
            stringBuilder.append(signals[i]).append(",");
        }
        stringBuilder.append(label);
        stringBuilder.append("\n");
        try {
            mBufferWriter.write(stringBuilder.toString());
            mBufferWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


}