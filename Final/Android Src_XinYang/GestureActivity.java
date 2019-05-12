package com.neilyxin.www.vibauth;

import android.content.SharedPreferences;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Environment;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.BufferedWriter;
import java.io.File;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.locks.ReentrantLock;

public class GestureActivity extends AppCompatActivity {

    private TextView txt_gesture_main;

    private Button btn_gesture_train;
    private Button btn_gesture_predict;
    private Button btn_gesture_reset;

    private ScrollView scroll_gesture;

    private Thread recordingThread;
    boolean shouldContinue = true;

    private static final int SAMPLE_RATE = 48000;
    private static final int SAMPLE_DURATION_MS = 100;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
//    private static final int SIGNAL_BUFFER_LENGTH_MS = 2000;
    byte[] recordingBuffer = new byte[RECORDING_LENGTH];
    int recordingOffset = 0;
    private final ReentrantLock recordingBufferLock = new ReentrantLock();
    private int startPoint;
    private static final int mNumberOfFFTPoints = 2048; // Must be the power of 2
    private Number[] frqDomain;
    //    private Number[] FFTresult;
    private static final int unit = 10;
    private static final int startHz = 17000;
    private static final int endHz = 19000;
    private static final int labelStepHz = 500;
    private AudioRecord record;
    private int bufferSize = 0;
    private static final int REQUEST_RECORD_AUDIO = 13;
    private final String TAG = getClass().getSimpleName();

    private static final String SVM_TRAIN = "SVM_train.txt";
    private static final String TF_FFT_DATASET = "TF_FFT_DATA.csv";
    private static final String TF_SIGNAL_DATASET = "TF_SIGNAL_DATA.csv";

//    private int currentLabel = 0;

    private static final int MAX_LABEL = 17;
    private int trainingDuration = 2000;

    private BufferedWriter svmBufferWriter;
    private BufferedWriter tfFFTBufferWriter;
    private BufferedWriter tfSignalBufferWriter;

    private ClassifierSVM classifierSVM;
    private ClassifierTFDNN classifierTFDNN;

    private StringBuilder txt_str;


    private String filepath;
    private String appFolderPath;


    private  MySignalProcessing mySignalProcessing;

    private byte[] lastSignal;
    private boolean shouldDetect = true;


    private FileHelper fileHelper;
//    private int labelsNum = 0;
    private MFCC mfcc;
    private AudioPlayer audioPlayer;
    private boolean PLAY_AUDIO_FROM_PHONE = false;
    private int trainCounter;

    private ArrayList<Float> DTWArray;

    private ArrayList<float[]> signalBufferList;
    private float[] referenceBuffer;
    private float[] signalBuffer;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_gesture);

        txt_gesture_main = findViewById(R.id.txt_gesture_main);
        btn_gesture_train = findViewById(R.id.btn_gesture_train);
        btn_gesture_predict = findViewById(R.id.btn_gesture_predict);
        btn_gesture_reset = findViewById(R.id.btn_gesture_reset);
        scroll_gesture = findViewById(R.id.scroll_gesture);

        mfcc = new MFCC();
//        fileHelper = new FileHelper();

        filepath = Environment.getExternalStorageDirectory().getPath();
        appFolderPath = filepath + "/VibAuth/";
        startPoint = startHz * mNumberOfFFTPoints / SAMPLE_RATE;

        mySignalProcessing = new MySignalProcessing(SAMPLE_RATE, mNumberOfFFTPoints, startPoint);

        classifierSVM = new ClassifierSVM(appFolderPath, mySignalProcessing,true,true);
        classifierTFDNN = new ClassifierTFDNN(true, true); // For data recording use only.

        txt_str = new StringBuilder();

//        txt_gesture_main.setText(txt_str.append("Press Button To Record Training Sets\n"));
//        txt_gesture_main.setText(txt_str.append("Preparing file...\n"));

        trainCounter = 0;

        audioPlayer = new AudioPlayer();
        DTWArray = new ArrayList<>();

        // Create file for training set
//        try {
//            File folder = new File(appFolderPath);
//            if(!folder.exists()){
//                folder.mkdirs();
//                txt_gesture_main.setText(txt_str.append("- Folder created.\n"));
//            } else {
//                txt_gesture_main.setText(txt_str.append("- Folder exists.\n"));
//            }
//
//
//            svmBufferWriter = fileHelper.getBufferWriter(appFolderPath + SVM_TRAIN);
//
//            if (svmBufferWriter != null) {
//                txt_gesture_main.setText(txt_str.append("- Training file created.\n"));
//            }
//
//
//            tfFFTBufferWriter = fileHelper.getBufferWriter(appFolderPath + TF_FFT_DATASET);
//            tfSignalBufferWriter = fileHelper.getBufferWriter(appFolderPath + TF_SIGNAL_DATASET);
//
//        } catch (Exception e) {
//            fileHelper.closeAllWriters();
//            txt_gesture_main.setText(txt_str.append("Error occurred when preparing training file.\n"));
//
//        }

//        txt_gesture_main.setText(txt_str.append("- File created successfully\nHold your finger to record Label.\n"));

        btn_gesture_train.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                trainCounter++;
                recordDataSet();
            }
        });

        btn_gesture_predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                trainCounter++;
                recordDataSet();
            }
        });


        btn_gesture_reset.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
//                SharedPreferences sharedPreferences = getSharedPreferences("LABELSNUM", MODE_PRIVATE);
//                SharedPreferences.Editor editor = sharedPreferences.edit();
//                editor.putInt("LABELSNUM", labelsNum);
//                editor.apply();
//
//                fileHelper.closeAllWriters();
//                String svmTrainOptions = "-t 0 "; // note the ending space
//                classifierSVM.train(svmTrainOptions);
                txt_gesture_main.setText("");
                txt_str = new StringBuilder();
                DTWArray.clear();
                signalBufferList.clear();
                signalBuffer = null;
                referenceBuffer = null;
                trainCounter = 0;
//                scroll_gesture.fullScroll(ScrollView.FOCUS_DOWN);
            }
        });


    }


    private void recordDataSet() {

        signalBufferList = new ArrayList<>();

        startRecording();
//        txt_train.setText(txt_str.append("Label " + currentLabel + " recording...\n"));
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                scroll_gesture.fullScroll(ScrollView.FOCUS_DOWN);
            }
        });

        TimerTask task = new TimerTask(){
            public void run(){
                //execute the task
                stopRecording();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
//                        txt_gesture_main.setText(txt_str.append("Label " + currentLabel + " recorded successfully.\n"));
//                        if (currentLabel < MAX_LABEL) {
//                            txt_train.setText(txt_str);
//                        }


                        scroll_gesture.fullScroll(ScrollView.FOCUS_DOWN);


                    }
                });
            }
        };

        Timer timer = new Timer();
        timer.schedule(task, trainingDuration);
    }




    //------------------------ Recording and Processing ---------------------------
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


        signalBuffer = new float[signalBufferList.get(0).length * signalBufferList.size()];
        for (int i = 0; i < signalBufferList.size(); i++) {
            for (int j = 0; j < signalBufferList.get(i).length; j++) {
//                Log.e(TAG, String.valueOf(signalBufferList.size() + " " + signalBufferList.size() * signalBufferList.get(i).length) +" " + signalBuffer.length);
                signalBuffer[i*signalBufferList.get(i).length + j] = signalBufferList.get(i)[j];
            }

        }
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (trainCounter == 1) {
                    referenceBuffer = signalBuffer;
                    txt_str.append("Training " + trainCounter + "complete.\n");
                    txt_gesture_main.setText(txt_str.toString());
                } else {
                    float DTW = mySignalProcessing.getDTW(referenceBuffer, signalBuffer);
                    txt_str.append("Training " + trainCounter + "complete. DTW distance: " + DTW + "\n");
                    txt_gesture_main.setText(txt_str.toString());
                }
            }
        });


        //TODO: Save recorded audio.
    }

    /**
     * Record task
     */
    private void record() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);
        // Estimate the buffer size we'll need for this device.
        bufferSize =
                AudioRecord.getMinBufferSize(
                        SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_8BIT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2;
        }
        byte[] audioBuffer = new byte[RECORDING_LENGTH];

        record =
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

            byte[] buffer = new byte[RECORDING_LENGTH];
            System.arraycopy(audioBuffer, 0, buffer, 0, RECORDING_LENGTH);
            double[] doubleBuffer = mySignalProcessing.getDoubleFromByte(buffer);
            Number[] FFTresult = mySignalProcessing.calculateFFT(audioBuffer);

            float[] FFTArray = new float[FFTresult.length];
            for (int i = 0; i < FFTresult.length; i++) {
                FFTArray[i] = Float.parseFloat(String.format("%.6f",FFTresult[i]));
//                stringBuilder.append(" " + (i+1) + ":" + new BigDecimal(String.format("%.6f",FFTresult[i])));
            }
//            float[] MFCCResult = mfcc.process(doubleBuffer);

            if (shouldContinue) {
                signalBufferList.add(FFTArray);
            }






//            classifierTFDNN.writeTensorFlowSignalTrainingData(audioBuffer, tfSignalBufferWriter, currentLabel);
//
//            Number[] FFTresult = mySignalProcessing.calculateFFT(audioBuffer);
//
//
//            if (FFTresult != null) {
//                classifierSVM.writeSVMTrainingData(FFTresult, MFCCResult, svmBufferWriter, currentLabel);
//                classifierTFDNN.writeTensorFlowFFTTrainingData(FFTresult, MFCCResult, tfFFTBufferWriter, currentLabel, mySignalProcessing);
//            }



        }
        record.stop();
        record.release();
    }

    @Override
    public void onPause() {
        stopRecording();
//        fileHelper.closeAllWriters();
        super.onPause();
    }
}
