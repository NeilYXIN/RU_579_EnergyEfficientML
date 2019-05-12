package com.neilyxin.www.vibauth;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Color;
import android.media.*;
import android.os.Build;
import android.os.Environment;
import android.os.Process;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RadioGroup;
import android.widget.TextView;

import com.androidplot.xy.BoundaryMode;
import com.androidplot.xy.LineAndPointFormatter;
import com.androidplot.xy.SimpleXYSeries;
import com.androidplot.xy.StepMode;
import com.androidplot.xy.XYPlot;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantLock;

public class MainActivity extends AppCompatActivity {
    private Button btn_plot;
    private Button btn_train;
    private Button btn_predict;
    private Button btn_gesture;
    private TextView txt_main;
    private TextView txt_auth;
    private RadioGroup radio_group_classifier;

    private Thread recordingThread;
    boolean shouldContinue = true;

    private static final int SAMPLE_RATE = 48000;
    private static final int SAMPLE_DURATION_MS = 200;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
    byte[] recordingBuffer = new byte[RECORDING_LENGTH];
    int recordingOffset = 0;
    private final ReentrantLock recordingBufferLock = new ReentrantLock();
    private int startPoint;
    private static final int mNumberOfFFTPoints =2048; // Must be the power of 2
    private Number[] frqDomain;
    private Number[] FFTresult;
    private static final int unit = 10;
    private static final int startHz = 17000;
    private static final int endHz = 19000;
    private static final int labelStepHz = 500;

    private XYPlot plot;
    private LineAndPointFormatter formatter;
    private int bufferSize = 0;
    private static final int REQUEST_RECORD_AUDIO = 13;
    private final String TAG = getClass().getSimpleName();

    private boolean isPredicting = false;
    private String filepath;
    private String appFolderPath;

    private ExecutorService predictionThreadPool;
    private MySignalProcessing mySignalProcessing;

    private ClassifierTFDNN classifierTFDNN;
    private ClassifierSVM classifierSVM;

    private static final int CLASSIFIER_SVM = 10;
    private static final int CLASSIFIER_TF_DNN = 11;

    private String DNN_MODEL_FILE;
    private String DNNmodelFileName = "VibAuthModel.pb";

    private int classifierId = 10;
    private boolean isTFDNNModelExist = false;
    private MFCC mfcc;

    private Authenticator authenticator;
    private static final int AUTHENTICATOR_QUEUE_SIZE = 6;
    private static final int AUTHENTICATOR_VALID_THRESHOLD = 5;
    private static final float AUTHENTICATOR_VALID_CONFIDENCE = (float) 0.70;
    private static final int AUTHENTICATION_INTERVAL_IN_MILLIS = 800;
    private StringBuilder inputStringBuilder;
    private long lastTime;

    private AudioPlayer audioPlayer;

    private boolean PLAY_AUDIO_FROM_PHONE = false;

    private List<Integer> patternPredictionList;

    boolean SVM_MFCC = false;
    boolean SVM_NORMALIZATION = false;
    boolean DNN_MFCC = false;
    boolean DNN_NORMALIZATION = true;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        requestMicrophonePermission();
        filepath = Environment.getExternalStorageDirectory().getPath();
        appFolderPath = filepath + "/VibAuth/";
        DNN_MODEL_FILE = appFolderPath + DNNmodelFileName;
        startPoint = startHz * mNumberOfFFTPoints / SAMPLE_RATE;
        mySignalProcessing = new MySignalProcessing(SAMPLE_RATE, mNumberOfFFTPoints, startPoint);
        authenticator = new Authenticator(AUTHENTICATOR_QUEUE_SIZE, AUTHENTICATOR_VALID_THRESHOLD);

        audioPlayer = new AudioPlayer();

        initializeWidgets();
        initializePlot();

        if (isTFDNNModelExist) {
            Log.d(TAG, "TFDNN MODEL EXIST");
            SharedPreferences sharedPreferences = getSharedPreferences("LABELSNUM", MODE_PRIVATE);
            classifierTFDNN = new ClassifierTFDNN(getAssets(),DNN_MODEL_FILE, sharedPreferences, DNN_MFCC, DNN_NORMALIZATION);
        }
    }

    //----------------------- Bind Widgets --------------------
    /**
     * Bind and configure widgets.
     */
    private void initializeWidgets() {
        btn_plot = findViewById(R.id.btn_plot);
        btn_train = findViewById(R.id.btn_train);
        btn_predict = findViewById(R.id.btn_predict);
        btn_gesture = findViewById(R.id.btn_gesture);
        txt_main = findViewById(R.id.txt_main);
        txt_auth = findViewById(R.id.txt_auth);
        radio_group_classifier = findViewById(R.id.radio_group_classifier);

        mfcc = new MFCC();
        classifierSVM = new ClassifierSVM(appFolderPath, mySignalProcessing, SVM_MFCC, SVM_NORMALIZATION);
        File SVMModel = new File(classifierSVM.getSVMModelName());
        if (SVMModel.exists()) {
            txt_main.setText("SVM Model file already exists, you can run prediction directly.");
        } else {
            txt_main.setText("No SVM Model file detected. Please run training first.");
        }

        File DNNModel = new File(DNN_MODEL_FILE);
        if (DNNModel.exists()) {
            isTFDNNModelExist = true;
            txt_main.setText("DNN Model file already exists, you can run prediction directly.");
        } else {
            isTFDNNModelExist =false;
            txt_main.setText("No DNN Model file detected. Please run training first.");
        }

        btn_plot.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (btn_plot.getText().equals("Plot")) {
                    btn_plot.setText("Stop");

                    startRecording();
                } else if (btn_plot.getText().equals("Stop")) {
                    btn_plot.setText("Plot");
                    stopRecording();
                }
            }
        });

        btn_train.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setClass(MainActivity.this, TrainActivity.class);
                startActivity(intent);
            }
        });

        btn_predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                File model = null;
                if (radio_group_classifier.getCheckedRadioButtonId() == R.id.radio_svm) {
                    classifierId = CLASSIFIER_SVM;
                    model = new File(classifierSVM.getSVMModelName());
                } else if (radio_group_classifier.getCheckedRadioButtonId() == R.id.radio_dnn) {
                    classifierId = CLASSIFIER_TF_DNN;
                    model = new File(DNN_MODEL_FILE);
                    if (model.exists()) {
                        isTFDNNModelExist = true;
                        SharedPreferences sharedPreferences = getSharedPreferences("LABELSNUM", MODE_PRIVATE);
                        classifierTFDNN = new ClassifierTFDNN(getAssets(),DNN_MODEL_FILE, sharedPreferences, DNN_MFCC, DNN_NORMALIZATION);
                    }
                }
                if (model.exists()) {
                    if (btn_predict.getText().equals("Predict")) {

                        patternPredictionList = new ArrayList<>();

                        authenticator.reset();
                        inputStringBuilder = new StringBuilder();
                        txt_auth.setText("");
                        predictionThreadPool = Executors.newFixedThreadPool(1);
                        btn_predict.setText("Stop");
                        isPredicting = true;
                        startRecording();
                    } else if (btn_predict.getText().equals("Stop")) {
                        StringBuilder sb = new StringBuilder();
                        for (int i: patternPredictionList) {
                            sb.append(i).append(",");
                        }
                        String s = sb.toString();
                        if (s.length() >= 1) {
                            System.out.println("PATTERN: " + s.substring(0, s.length() - 1));
                        }
                        btn_predict.setText("Predict");
                        isPredicting = false;
                        stopRecording();
                        predictionThreadPool.shutdownNow();
                    }
                }
            }
        });


        btn_gesture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setClass(MainActivity.this, GestureActivity.class);
                startActivity(intent);
            }
        });

    }

    //---------------------------- Graph -------------------------------
    /**
     * Initialize Androidplot Widget.
     */
    private void initializePlot() {
        frqDomain = new Number[mNumberOfFFTPoints / 2 - startPoint];
        for(int i = 0; i < (mNumberOfFFTPoints / 2 - startPoint); i++){
            frqDomain[i] = SAMPLE_RATE * (i + startPoint) / (unit * mNumberOfFFTPoints);
        }
        // initialize our XYPlot reference:
        plot = findViewById(R.id.plot);
//        plot.setRenderMode(Plot.RenderMode.USE_MAIN_THREAD);
        // Set line color: red, vertex color: null, fill color: blue
        formatter = new LineAndPointFormatter(Color.RED,null, null, null);
        formatter.setLegendIconEnabled(false);
        //plot.addSeries(seriesData, formatter);
        plot.setRangeBoundaries(0, 50, BoundaryMode.GROW);
        plot.setDomainBoundaries(startHz/unit, endHz/unit, BoundaryMode.FIXED);
        plot.setDomainStep(StepMode.INCREMENT_BY_VAL, labelStepHz/unit);
        plot.setLinesPerDomainLabel(5);
//        plot.getGraph().getDomainGridLinePaint().setColor(Color.TRANSPARENT);
//        //set all domain lines to transperent
//
//        plot.getGraph().getRangeSubGridLinePaint().setColor(Color.TRANSPARENT);
//        //set all range lines to transperent
//
//        plot.getGraph().getRangeGridLinePaint().setColor(Color.TRANSPARENT);
//        //set all sub range lines to transperent
//
//        plot.getGraph().getDomainSubGridLinePaint().setColor(Color.TRANSPARENT);
//        //set all sub domain lines to transperent
//        plot.getGraph().getBackgroundPaint().setColor(Color.WHITE);
    }

    /**
     * Update the graph when gets new data
     */
    private synchronized void updatePlot(Number[] seriesDataNumbers) {
        plot.clear();
        plot.addSeries(new SimpleXYSeries(Arrays.asList(frqDomain), Arrays.asList(seriesDataNumbers), "seriesData"), formatter);
        plot.redraw();
    }

    //-------------------------- Permission ------------------------
    /**
     * Request audio recording permission.
     */
    private void requestMicrophonePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(
                    new String[]{android.Manifest.permission.RECORD_AUDIO, android.Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.INTERNET}, REQUEST_RECORD_AUDIO);
        }
    }

    /**
     * Get permission returned result.
     * @param requestCode
     * @param permissions
     * @param grantResults
     */
    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_RECORD_AUDIO
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
        }
    }

    //------------------------ Recording and Recognition ---------------------------
    /**
     * Start record thread
     */
    public synchronized void startRecording() {
        if (PLAY_AUDIO_FROM_PHONE) {
            audioPlayer.play(true);
        }
        if (recordingThread != null) {
            return;
        }
        shouldContinue = true;
        recordingThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                record();
                            }
                        });
        recordingThread.start();
    }

    /**
     * Stop record thread
     */
    public synchronized void stopRecording() {
        if (PLAY_AUDIO_FROM_PHONE) {
            audioPlayer.pause();
        }
        if (recordingThread == null) {
            return;
        }
        shouldContinue = false;
        recordingThread = null;
        Log.v(TAG, "Stop recording");
        //TODO: Save recorded audio.
    }

    /**
     * Record task
     */
    private void record() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
        // Estimate the buffer size we'll need for this device.
        bufferSize =
                AudioRecord.getMinBufferSize(
                        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_8BIT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2;
        }

        byte[] audioBuffer = new byte[RECORDING_LENGTH];
        AudioRecord record =
                new AudioRecord(
                        MediaRecorder.AudioSource.UNPROCESSED,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufferSize);
        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "Audio Record can't initialize!");
            return;
        }
        record.startRecording();
        Log.v(TAG, "Start recording");

        while (shouldContinue) {
            int numberRead = record.read(audioBuffer, 0, audioBuffer.length);

            FFTresult = mySignalProcessing.calculateFFT(audioBuffer);
            if (FFTresult != null) {
                final Number[] seriesDataNumbers = mySignalProcessing.window(FFTresult);

                updatePlot(seriesDataNumbers);

                byte[] buffer = new byte[RECORDING_LENGTH];
                System.arraycopy(audioBuffer, 0, buffer, 0, RECORDING_LENGTH);
                double[] doubleBuffer = mySignalProcessing.getDoubleFromByte(buffer);
                final float[] MFCCResult = mfcc.process(doubleBuffer);

                //TODO: Put SVM recognition methods here.
                if (isPredicting) {
                    predictionThreadPool.execute(new Runnable() {
                        @Override
                        public void run() {
                            int predictionLabel = -1;
                            float predictionConfidence = -1;
                            switch (classifierId){
                                case CLASSIFIER_SVM: {
                                    predictionLabel = classifierSVM.predict(seriesDataNumbers, MFCCResult, txt_main);
                                    break;
                                }
                                case CLASSIFIER_TF_DNN: {
                                    float[] outputs = classifierTFDNN.predict(seriesDataNumbers, MFCCResult, mySignalProcessing);
                                    predictionLabel = (int)outputs[0];
                                    predictionConfidence = outputs[1];
                                    break;
                                }
                            }

                            if (predictionLabel != 0) {
                                patternPredictionList.add(predictionLabel);
                            }

                            final int finalPredictionLabel = predictionLabel;
                            final float finalPredictionConfidence = predictionConfidence;
                            authenticator.add(finalPredictionLabel, finalPredictionConfidence);
//                            authenticator.printPredictionQueue();
                            final boolean isAuthValid = authenticator.getMode();

                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    // Display authentication result
//                                    if (isAuthValid) {
//                                        if (authenticator.getModeLabel() == 0) {
////                                                txt_auth.setText("");
//                                        } else {
//                                            if (classifierId == CLASSIFIER_SVM || (classifierId == CLASSIFIER_TF_DNN && authenticator.getModeConfidence() > AUTHENTICATOR_VALID_CONFIDENCE)) {
//                                                long time = System.currentTimeMillis();
//                                                if (time - lastTime > AUTHENTICATION_INTERVAL_IN_MILLIS) {
//                                                    inputStringBuilder.append(authenticator.getModeLabel() + " ");
//                                                    txt_auth.setText(inputStringBuilder.toString());
//                                                    lastTime = time;
//                                                }
////                                                    txt_auth.setText("Auth Label: " + authenticator.getModeLabel() + "\nAuth Confidence: " + authenticator.getModeConfidence());
//                                            } else {
////                                                    txt_auth.setText("INVALID");
//                                            }
//                                        }
//                                    } else {
////                                            txt_auth.setText("");
//                                    }
//
//                                    // Display real-time prediction result and confidence
//                                    if (finalPredictionLabel == 0) {
//                                        txt_main.setText("Prediction: Empty\nConfidence: " + String.format("%.2f", finalPredictionConfidence));
//                                    } else {
//                                        txt_main.setText("Prediction: " + finalPredictionLabel + "\nConfidence: " + String.format("%.2f", finalPredictionConfidence));
//                                    }

                                    updateText(isAuthValid, finalPredictionLabel, finalPredictionConfidence);
                                }
                            });
                        }
                    });
                }
            }
        }
        record.stop();
        record.release();
    }

    //------------------- System Configuration ---------------------------

    private synchronized void updateText(boolean isAuthValid, int finalPredictionLabel, float finalPredictionConfidence) {
        if (isAuthValid) {
            if (authenticator.getModeLabel() == 0) {
//                                                txt_auth.setText("");
            } else {
                if (classifierId == CLASSIFIER_SVM || (classifierId == CLASSIFIER_TF_DNN && authenticator.getModeConfidence() > AUTHENTICATOR_VALID_CONFIDENCE)) {
                    long time = System.currentTimeMillis();
                    if (time - lastTime > AUTHENTICATION_INTERVAL_IN_MILLIS) {
                        inputStringBuilder.append(authenticator.getModeLabel() + " ");
                        txt_auth.setText(inputStringBuilder.toString());
                        lastTime = time;
                    }
//                                                    txt_auth.setText("Auth Label: " + authenticator.getModeLabel() + "\nAuth Confidence: " + authenticator.getModeConfidence());
                } else {
//                                                    txt_auth.setText("INVALID");
                }
            }
        } else {
//                                            txt_auth.setText("");
        }

        // Display real-time prediction result and confidence
        if (finalPredictionLabel == 0) {
            txt_main.setText("Prediction: Empty\nConfidence: " + String.format("%.2f", finalPredictionConfidence));
        } else {
            txt_main.setText("Prediction: " + finalPredictionLabel + "\nConfidence: " + String.format("%.2f", finalPredictionConfidence));
        }
    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
    }

}