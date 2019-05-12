package com.neilyxin.www.vibauth;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * This is a customized class to do some signal processing tasks.
 * 1. Calculating the FFT
 * 2. Get the windowed FFT
 */
class MySignalProcessing {
    private int SAMPLE_RATE;
    private int mNumberOfFFTPoints; // Must be the power of 2
    private int startPoint;

    private long mean;
    private long sigma;
    private DynamicTimeWrapping dynamicTimeWrapping;

    MySignalProcessing(int SAMPLE_RATE, int mNumberOfFFTPoints, int startPoint) {
        this.SAMPLE_RATE = SAMPLE_RATE;
        this.mNumberOfFFTPoints = mNumberOfFFTPoints;
        this.startPoint = startPoint;
        this.dynamicTimeWrapping = new DynamicTimeWrapping();
//        getThresholdLocal(mContext);
    }

    /**
     * Calculate FFT.
     * @param signal
     * @return
     */
    Number[] calculateFFT(byte[] signal)
    {
        double mMaxFFTSample;
        double temp;
        Complex[] y;
        Complex[] complexSignal = new Complex[mNumberOfFFTPoints];
        Number[] absSignal = new Number[mNumberOfFFTPoints/2];

        for(int i = 0; i < mNumberOfFFTPoints; i++){
            // ? change byte to double?
            temp = (double)((signal[2*i] & 0xFF) | (signal[2*i+1] << 8)) / 32768.0F;
            complexSignal[i] = new Complex(temp,0.0);
        }

        y = FFT.fft(complexSignal); // --> Here I use FFT class

        mMaxFFTSample = 0.0;
        int mPeakPos = 0;
        for(int i = 0; i < (mNumberOfFFTPoints/2); i++)
        {
            absSignal[i] = Math.sqrt(Math.pow(y[i].re(), 2) + Math.pow(y[i].im(), 2));
            if(absSignal[i].doubleValue() > mMaxFFTSample)
            {
                mMaxFFTSample = absSignal[i].doubleValue();
                mPeakPos = i;
            }
            int MAXfrq = mPeakPos * SAMPLE_RATE / mNumberOfFFTPoints;
        }
        return absSignal;
    }

    /**
     * Show data from startHz to endHz.
     * @param signal
     * @return
     */
    Number[] window(Number[] signal){
        Number[] windowAbsSignal;
        windowAbsSignal = new Number[mNumberOfFFTPoints / 2 - startPoint];
        for(int i = 0; i < (mNumberOfFFTPoints / 2 - startPoint); i++){
            windowAbsSignal[i] = signal[startPoint + i];
        }
        return windowAbsSignal;
    }

    long getShortTermEnergy(byte[] signal) {
        long shortTermEnergy = 0;
        for (byte amplitude: signal) {
            shortTermEnergy += amplitude * amplitude;
        }
        return shortTermEnergy;
    }


    long getMean(List<Long> shortTermEnergyList) {
        long sum = 0;
        for (long shortTermEnergy: shortTermEnergyList) {
            sum+=shortTermEnergy;
        }
        return sum/shortTermEnergyList.size();
    }

    long getStdDev(List<Long> shortTermEnergyList, long mean) {
        long stdDevSum = 0;
        for (long shortTermEnergy: shortTermEnergyList) {
            stdDevSum += (shortTermEnergy - mean) * (shortTermEnergy - mean);
        }
        return  (long) Math.sqrt(stdDevSum/shortTermEnergyList.size());
    }

//    public void saveThresholdLocal(long mean, long sigma, Context mContext) {
//        SharedPreferences sharedPreferences = mContext.getSharedPreferences("THRESHOLD", MODE_PRIVATE);
//        SharedPreferences.Editor editor = sharedPreferences.edit();
//        editor.putLong("MEAN", mean);
//        editor.putLong("SIGMA", sigma);
//        editor.apply();
//        getThresholdLocal(mContext);
//    }

//    public void getThresholdLocal(Context mContext) {
//        SharedPreferences sharedPreferences = mContext.getSharedPreferences("THRESHOLD", MODE_PRIVATE);
//        sharedPreferences.getLong("MEAN", -1);
//        sharedPreferences.getLong("SIGMA", -1);
//    }

    long getMean() {
        return mean;
    }

    long getSigma() {
        return sigma;
    }


    double[] getDoubleFromByte(byte[] bytes) {
        double[] out = new double[bytes.length / 2]; // will drop last byte if odd number
        ByteBuffer bb = ByteBuffer.wrap(bytes);
        for (int i = 0; i < out.length; i++) {
            out[i] = bb.getShort()/32767.0;
        }
        return out;
    }

    float getDTW(float[] sequenceA, float[] sequenceB) {
        float DTW = dynamicTimeWrapping.getDTW(sequenceA, sequenceB);
//        this.dynamicTimeWrapping = new DynamicTimeWrapping();
        return DTW;
    }

    float[] normalize(float[] signals) {
        float[] normalizedSignals = new float[signals.length];
        float max = Float.MIN_VALUE;
        float min = Float.MAX_VALUE;
        for (float signal : signals) {
            max = Math.max(max, signal);
            min = Math.min(min, signal);
        }
        for (int i = 0; i < signals.length; i++) {
            normalizedSignals[i] = (signals[i]-min) / (max - min);
        }
        return normalizedSignals;
    }

    float[] normalize(Number[] signals) {
        float[] floatFFTResult = new float[signals.length];
        for (int i = 0; i < signals.length; i++) {
            floatFFTResult[i] = signals[i].floatValue();
        }

        float[] normalizedSignals = new float[floatFFTResult.length];
        float max = Float.MIN_VALUE;
        float min = Float.MAX_VALUE;
        for (float signal : floatFFTResult) {
            max = Math.max(max, signal);
            min = Math.min(min, signal);
        }
        for (int i = 0; i < signals.length; i++) {
            normalizedSignals[i] = (floatFFTResult[i]-min) / (max - min);
        }
        return normalizedSignals;
    }

}