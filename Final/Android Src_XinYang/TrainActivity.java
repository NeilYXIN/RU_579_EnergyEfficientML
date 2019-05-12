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
import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.locks.ReentrantLock;

public class TrainActivity extends AppCompatActivity {
    private TextView txt_train;
    private Button btn_label1;
    private Button btn_label2;
    private Button btn_label3;
    private Button btn_label4;
    private Button btn_label5;
    private Button btn_label6;
    private Button btn_label7;
    private Button btn_label8;
    private Button btn_label9;
    private Button btn_label10;
    private Button btn_label11;
    private Button btn_label12;
    private Button btn_label13;
    private Button btn_label14;
    private Button btn_label15;
    private Button btn_label16;

    private Button btn_label_empty;
    private Button btn_stop;
    private ScrollView scrollView;

    private Thread recordingThread;
    boolean shouldContinue = true;

    private static final int SAMPLE_RATE = 48000;
    private static final int SAMPLE_DURATION_MS = 200;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
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

    private int currentLabel = 0;

    private static final int MAX_LABEL = 17;
    private int trainingDuration = 5000;

    private BufferedWriter svmBufferWriter;
    private BufferedWriter tfFFTBufferWriter;
    private BufferedWriter tfSignalBufferWriter;

    private ClassifierSVM classifierSVM;

    private StringBuilder txt_str;


    private String filepath;
    private String appFolderPath;


    private  MySignalProcessing mySignalProcessing;

    private byte[] lastSignal;
    private boolean shouldDetect = true;


    private FileHelper fileHelper;
    private int labelsNum = 0;
    private MFCC mfcc;
    private AudioPlayer audioPlayer;
    private boolean PLAY_AUDIO_FROM_PHONE = false;

    private ClassifierTFDNN classifierTFDNN;

    boolean SVM_MFCC = false;
    boolean SVM_NORMALIZATION = false;
    boolean DNN_MFCC = false;
    boolean DNN_NORMALIZATION = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_train);

        txt_train = findViewById(R.id.txt_train);
        btn_label1 = findViewById(R.id.btn_label1);
        btn_label2 = findViewById(R.id.btn_label2);
        btn_label3 = findViewById(R.id.btn_label3);
        btn_label4 = findViewById(R.id.btn_label4);
        btn_label5 = findViewById(R.id.btn_label5);
        btn_label6 = findViewById(R.id.btn_label6);
        btn_label7 = findViewById(R.id.btn_label7);
        btn_label8 = findViewById(R.id.btn_label8);
        btn_label9 = findViewById(R.id.btn_label9);
        btn_label10 = findViewById(R.id.btn_label10);
        btn_label11 = findViewById(R.id.btn_label11);
        btn_label12 = findViewById(R.id.btn_label12);
        btn_label13 = findViewById(R.id.btn_label13);
        btn_label14 = findViewById(R.id.btn_label14);
        btn_label15 = findViewById(R.id.btn_label15);
        btn_label16 = findViewById(R.id.btn_label16);


        btn_label_empty = findViewById(R.id.btn_label_empty);
        btn_stop = findViewById(R.id.btn_stop);
        scrollView = findViewById(R.id.scroll_train);

        mfcc = new MFCC();
        fileHelper = new FileHelper();

        filepath = Environment.getExternalStorageDirectory().getPath();
        appFolderPath = filepath + "/VibAuth/";
        startPoint = startHz * mNumberOfFFTPoints / SAMPLE_RATE;

        mySignalProcessing = new MySignalProcessing(SAMPLE_RATE, mNumberOfFFTPoints, startPoint);

        classifierSVM = new ClassifierSVM(appFolderPath, mySignalProcessing, SVM_MFCC, SVM_NORMALIZATION);
        classifierTFDNN = new ClassifierTFDNN(DNN_MFCC, DNN_NORMALIZATION); // For data recording use only.

        txt_str = new StringBuilder();
        txt_train.setText(txt_str.append("Press Button To Record Training Sets\n"));
        txt_train.setText(txt_str.append("Preparing file...\n"));

        labelsNum = 0;

        audioPlayer = new AudioPlayer();

        // Create file for training set
        try {
            File folder = new File(appFolderPath);
            if(!folder.exists()){
                folder.mkdirs();
                txt_train.setText(txt_str.append("- Folder created.\n"));
            } else {
                txt_train.setText(txt_str.append("- Folder exists.\n"));
            }


            svmBufferWriter = fileHelper.getBufferWriter(appFolderPath + SVM_TRAIN);

            if (svmBufferWriter != null) {
                txt_train.setText(txt_str.append("- Training file created.\n"));
            }


            tfFFTBufferWriter = fileHelper.getBufferWriter(appFolderPath + TF_FFT_DATASET);
            tfSignalBufferWriter = fileHelper.getBufferWriter(appFolderPath + TF_SIGNAL_DATASET);

        } catch (Exception e) {
            fileHelper.closeAllWriters();
            txt_train.setText(txt_str.append("Error occurred when preparing training file.\n"));

        }

        txt_train.setText(txt_str.append("- File created successfully\nHold your finger to record Label.\n"));

        btn_label1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 1;
                recordDataSet();
            }
        });

        btn_label2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 2;
                recordDataSet();
            }
        });

        btn_label3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 3;
                recordDataSet();
            }
        });

        btn_label4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 4;
                recordDataSet();
            }
        });

        btn_label5.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 5;
                recordDataSet();
            }
        });

        btn_label6.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 6;
                recordDataSet();
            }
        });

        btn_label7.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 7;
                recordDataSet();
            }
        });

        btn_label8.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 8;
                recordDataSet();
            }
        });

        btn_label9.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 9;
                recordDataSet();
            }
        });

        btn_label10.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 10;
                recordDataSet();
            }
        });

        btn_label11.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 11;
                recordDataSet();
            }
        });

        btn_label12.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 12;
                recordDataSet();
            }
        });

        btn_label13.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 13;
                recordDataSet();
            }
        });

        btn_label14.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 14;
                recordDataSet();
            }
        });

        btn_label15.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 15;
                recordDataSet();
            }
        });

        btn_label16.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 16;
                recordDataSet();
            }
        });

        btn_label_empty.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                currentLabel = 0;
                recordDataSet();
            }
        });

        btn_stop.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                SharedPreferences sharedPreferences = getSharedPreferences("LABELSNUM", MODE_PRIVATE);
                SharedPreferences.Editor editor = sharedPreferences.edit();
                editor.putInt("LABELSNUM", labelsNum);
                editor.apply();

                fileHelper.closeAllWriters();
                String svmTrainOptions = "-t 0 "; // note the ending space
                classifierSVM.train(svmTrainOptions);
                txt_train.setText(txt_str.append("Model training complete.\n"));
                scrollView.fullScroll(ScrollView.FOCUS_DOWN);
            }
        });
    }

    private void recordDataSet() {
        labelsNum++;

        startRecording();
        txt_train.setText(txt_str.append("Label " + currentLabel + " recording...\n"));
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                scrollView.fullScroll(ScrollView.FOCUS_DOWN);
            }
        });

        TimerTask task = new TimerTask(){
            public void run(){
                //execute the task
                stopRecording();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        txt_train.setText(txt_str.append("Label " + currentLabel + " recorded successfully.\n"));
                        if (currentLabel < MAX_LABEL) {
                            txt_train.setText(txt_str);
                        }

                        scrollView.fullScroll(ScrollView.FOCUS_DOWN);


                    }
                });
            }
        };

        Timer timer = new Timer();
        timer.schedule(task, trainingDuration);
    }


//    /**
//     * Update the graph when gets new data
//     */
//    private void writeTensorFlowFFTTrainingData(Number[] FFTresult, float[] MFCCResult, BufferedWriter mBufferWriter, int label) {
//        Number[] seriesDataNumbers = mySignalProcessing.window(FFTresult);
//        if (seriesDataNumbers[0].doubleValue() != 0.0) {
//            StringBuilder stringBuilder = new StringBuilder();
//            for (int i = 0; i < seriesDataNumbers.length; i++) {
//                stringBuilder.append(new BigDecimal(String.format("%.6f",seriesDataNumbers[i]))).append(",");
//
//            }
//
//            // MFCC data
//            for (int i = seriesDataNumbers.length; i < seriesDataNumbers.length + MFCCResult.length; i++) {
//                stringBuilder.append(String.format("%.6f",MFCCResult[i-seriesDataNumbers.length])).append(",");
//            }
//
//            stringBuilder.append(label);
//            stringBuilder.append("\n");
//            try {
//                mBufferWriter.write(stringBuilder.toString());
//                mBufferWriter.flush();
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }
//    }
//
//
//    /**
//     * Update the graph when gets new data
//     */
//    private void writeTensorFlowSignalTrainingData(byte[] signals, BufferedWriter mBufferWriter, int label) {
//        StringBuilder stringBuilder = new StringBuilder();
//        for (int i = 0; i < signals.length; i++) {
//            stringBuilder.append(signals[i]).append(",");
//        }
//        stringBuilder.append(label);
//        stringBuilder.append("\n");
//        try {
//            mBufferWriter.write(stringBuilder.toString());
//            mBufferWriter.flush();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//    }

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

            float[] MFCCResult = mfcc.process(doubleBuffer);

            classifierTFDNN.writeTensorFlowSignalTrainingData(audioBuffer, tfSignalBufferWriter, currentLabel);

            Number[] FFTresult = mySignalProcessing.calculateFFT(audioBuffer);


            if (FFTresult != null) {
                classifierSVM.writeSVMTrainingData(FFTresult, MFCCResult, svmBufferWriter, currentLabel);
                classifierTFDNN.writeTensorFlowFFTTrainingData(FFTresult, MFCCResult, tfFFTBufferWriter, currentLabel, mySignalProcessing);
            }
        }
        record.stop();
        record.release();
    }

    @Override
    public void onPause() {
        stopRecording();
        fileHelper.closeAllWriters();
        super.onPause();
    }

}