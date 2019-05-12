package com.neilyxin.www.vibauth;

import android.media.AudioManager;
import android.media.MediaPlayer;
import android.os.Environment;
import android.util.Log;

import java.io.IOException;

class AudioPlayer {
    private MediaPlayer mediaPlayer;
    private final String signalFileName = "signal.wav";
    private Thread playerThread;
    private final String TAG = this.getClass().getSimpleName();

    AudioPlayer() {
        mediaPlayer = new MediaPlayer();
        String filepath = Environment.getExternalStorageDirectory().getPath();
        String appFolderPath = filepath + "/VibAuth/";
        try {
            mediaPlayer.setDataSource(appFolderPath + signalFileName);
            mediaPlayer.prepareAsync();
        } catch (IOException e) {
            e.printStackTrace();
        }


    }

    void play(final boolean looping) {
        playerThread = new Thread(new Runnable() {
            @Override
            public void run() {
                mediaPlayer.setLooping(looping);
                mediaPlayer.start();
            }
        });

        playerThread.start();

    }

    void pause() {
        if (mediaPlayer.isPlaying()) {
            mediaPlayer.pause();
            Log.d(TAG, "MediaPlayer paused!!");
        }
        if (playerThread != null) {
            playerThread.interrupt();
        }
    }

    void stop() {
        mediaPlayer.stop();
    }

//    void release() {
//        mediaPlayer.release();
//    }
//
//    void reset() {
//        mediaPlayer.reset();
//    }




}
