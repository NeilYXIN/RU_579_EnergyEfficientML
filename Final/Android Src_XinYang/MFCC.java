package com.neilyxin.www.vibauth;

/*
    Turn Librosa Mfcc feature into Java code.
	Parameters are set to the librosa default for the purpose of android demo.
	The FFT code is taken from org.ioe.tprsa.audio.feature.
 */

/**
 * Mel-Frequency Cepstral Coefficients based on LibROSA
 *
 * @author Xin Yang
 *
 */


class MFCC {

    private final static int       n_mfcc       		= 40; // Number of features extracted
    private final static double    fMin                 = 0.0;
    private final static int       n_fft                = 2048;
    private final static int       hop_length           = 512;
    private final static int	   n_mels               = 128;
    private final static double    sampleRate           = 16000.0;
    private final static double    fMax                 = sampleRate/2.0;

    MFCCFFT fft = new MFCCFFT();

    float[] process(double[] doubleInputBuffer) {
        final double[][] mfccResult = dctMfcc(doubleInputBuffer);
        return finalshape(mfccResult);
    }

    //MFCC into 1d
    private float[] finalshape(double[][] mfccSpecTro){
        float[] finalMfcc = new float[mfccSpecTro.length];
        for (int i = 0; i < mfccSpecTro.length; i++){
            float sum = 0;
            for (int j = 0; j < mfccSpecTro[i].length; j++){
                sum += mfccSpecTro[i][j];
            }
            float mean = sum/mfccSpecTro[i].length;
            finalMfcc[i] = mean;
        }
        return finalMfcc;
    }

    // DCT to MFCC
    private double[][] dctMfcc(double[] y){
        final double[][] specTroGram = powerToDb(melSpectrogram(y));
        final double[][] dctBasis = dctFilter(n_mfcc, n_mels);
        double[][] mfccSpecTro = new double[n_mfcc][specTroGram[0].length];
        for (int i = 0; i < n_mfcc; i++){
            for (int j = 0; j < specTroGram[0].length; j++){
                for (int k = 0; k < specTroGram.length; k++){
                    mfccSpecTro[i][j] += dctBasis[i][k]*specTroGram[k][j];
                }
            }
        }
        return mfccSpecTro;
    }


    // Mel Spectrogram
    private double[][] melSpectrogram(double[] y){
        double[][] melBasis = melFilter();
        double[][] spectro = stftMagSpec(y);
        double[][] melS = new double[melBasis.length][spectro[0].length];
        for (int i = 0; i < melBasis.length; i++){
            for (int j = 0; j < spectro[0].length; j++){
                for (int k = 0; k < melBasis[0].length; k++){
                    melS[i][j] += melBasis[i][k]*spectro[k][j];
                }
            }
        }
        return melS;
    }

    //Short-time Fourier transform (STFT)
    private double[][] stftMagSpec(double[] y){
        final double[] fftwin = getWindow();
        double[] ypad = new double[n_fft+y.length];
        for (int i = 0; i < n_fft/2; i++){
            ypad[(n_fft/2)-i-1] = y[i+1];
            ypad[(n_fft/2)+y.length+i] = y[y.length-2-i];
        }
        for (int j = 0; j < y.length; j++){
            ypad[(n_fft/2)+j] = y[j];
        }
        final double[][] frame = yFrame(ypad);
        double[][] fftmagSpec = new double[1+n_fft/2][frame[0].length];
        double[] fftFrame = new double[n_fft];
        for (int k = 0; k < frame[0].length; k++){
            for (int l =0; l < n_fft; l++){
                fftFrame[l] = fftwin[l]*frame[l][k];
            }
            double[] magSpec = magSpectrogram(fftFrame);
            for (int i =0; i < 1+n_fft/2; i++){
                fftmagSpec[i][k] = magSpec[i];
            }
        }
        return fftmagSpec;
    }

    private double[] magSpectrogram(double[] frame){
        double[] magSpec = new double[frame.length];
        fft.process(frame);
        for (int m = 0; m < frame.length; m++) {
            magSpec[m] = fft.real[m] * fft.real[m] + fft.imag[m] * fft.imag[m];
        }
        return magSpec;
    }

    // Hanning Window
    private double[] getWindow(){
        double[] win = new double[n_fft];
        for (int i = 0; i < n_fft; i++){
            win[i] = 0.5 - 0.5 * Math.cos(2.0*Math.PI*i/n_fft);
        }
        return win;
    }

    // Frame
    private double[][] yFrame(double[] ypad){
        final int n_frames = 1 + (ypad.length - n_fft) / hop_length;
        double[][] winFrames = new double[n_fft][n_frames];
        for (int i = 0; i < n_fft; i++){
            for (int j = 0; j < n_frames; j++){
                winFrames[i][j] = ypad[j*hop_length+i];
            }
        }
        return winFrames;
    }

    // Power to dB
    private double[][] powerToDb(double[][] melS){
        double[][] log_spec = new double[melS.length][melS[0].length];
        double maxValue = -100;
        for (int i = 0; i < melS.length; i++){
            for (int j = 0; j < melS[0].length; j++){
                double magnitude = Math.abs(melS[i][j]);
                if (magnitude > 1e-10){
                    log_spec[i][j]=10.0*log10(magnitude);
                }else{
                    log_spec[i][j]=10.0*(-10);
                }
                if (log_spec[i][j] > maxValue){
                    maxValue = log_spec[i][j];
                }
            }
        }
        for (int i = 0; i < melS.length; i++){
            for (int j = 0; j < melS[0].length; j++){
                if (log_spec[i][j] < maxValue - 80.0){
                    log_spec[i][j] = maxValue - 80.0;
                }
            }
        }
        return log_spec;
    }

    // Dct
    private double[][] dctFilter(int n_filters, int n_input){
        //Discrete cosine transform (DCT type-III) basis.
        double[][] basis = new double[n_filters][n_input];
        double[] samples = new double[n_input];
        for (int i = 0; i < n_input; i++){
            samples[i] = (1 + 2*i) * Math.PI/(2.0*(n_input));
        }
        for (int j = 0; j < n_input; j++){
            basis[0][j] = 1.0/Math.sqrt(n_input);
        }
        for (int i = 1; i < n_filters; i++){
            for (int j = 0; j < n_input; j++){
                basis[i][j] = Math.cos(i*samples[j]) * Math.sqrt(2.0/(n_input));
            }
        }
        return basis;
    }

    // Mel
    private double[][] melFilter(){
        final double[] fftFreqs = fftFreq();
        final double[] melF = melFreq(n_mels+2);
        double[] fdiff = new double[melF.length-1];
        for (int i = 0; i < melF.length-1; i++){
            fdiff[i] = melF[i+1]-melF[i];
        }
        double[][] ramps = new double[melF.length][fftFreqs.length];
        for (int i = 0; i < melF.length; i++){
            for (int j = 0; j < fftFreqs.length; j++){
                ramps[i][j] = melF[i]-fftFreqs[j];
            }
        }
        double[][] weights = new double[n_mels][1+n_fft/2];
        for (int i = 0; i < n_mels; i++){
            for (int j = 0; j < fftFreqs.length; j++){
                double lowerF = -ramps[i][j] / fdiff[i];
                double upperF = ramps[i+2][j] / fdiff[i+1];
                if (lowerF > upperF && upperF>0){
                    weights[i][j] = upperF;
                }else if (lowerF > upperF && upperF<0){
                    weights[i][j] = 0;
                }else if (lowerF < upperF && lowerF>0){
                    weights[i][j] =lowerF;
                }else if (lowerF < upperF && lowerF<0){
                    weights[i][j] = 0;
                }else {}
            }
        }
        double enorm[] = new double[n_mels];
        for (int i = 0; i < n_mels; i++){
            enorm[i] = 2.0 / (melF[i+2]-melF[i]);
            for (int j = 0; j < fftFreqs.length; j++){
                weights[i][j] *= enorm[i];
            }
        }
        return weights;
    }

    // FFT frequencies
    private double[] fftFreq() {
        double[] freqs = new double[1+n_fft/2];
        for (int i = 0; i < 1+n_fft/2; i++){
            freqs[i] = 0 + (sampleRate/2)/(n_fft/2) * i;
        }
        return freqs;
    }

    // Mel frequencies
    private double[] melFreq(int numMels) {
        double[] LowFFreq = new double[1];
        double[] HighFFreq = new double[1];
        LowFFreq[0] = fMin;
        HighFFreq[0] = fMax;
        final double[] melFLow    = freqToMel(LowFFreq);
        final double[] melFHigh   = freqToMel(HighFFreq);
        double[] mels = new double[numMels];
        for (int i = 0; i < numMels; i++) {
            mels[i] = melFLow[0] + (melFHigh[0] - melFLow[0]) / (numMels-1) * i;
        }
        return melToFreq(mels);
    }

    // Mel to Hz, htk
    private double[] melToFreqS(double[] mels) {
        double[] freqs = new double[mels.length];
        for (int i = 0; i < mels.length; i++) {
            freqs[i] = 700.0 * (Math.pow(10, mels[i]/2595.0) - 1.0);
        }
        return freqs;
    }


    // Hz to Mel, htk
    protected double[] freqToMelS(double[] freqs) {
        double[] mels = new double[freqs.length];
        for (int i = 0; i < freqs.length; i++){
            mels[i] = 2595.0 * log10(1.0 + freqs[i]/700.0);
        }
        return mels;
    }

    // Mel to Hz, Slaney
    private double[] melToFreq(double[] mels) {
        final double f_min = 0.0;
        final double f_sp = 200.0 / 3;
        double[] freqs = new double[mels.length];
        final double min_log_hz = 1000.0;                         // beginning of log region (Hz)
        final double min_log_mel = (min_log_hz - f_min) / f_sp;  // same (Mels)
        final double logstep = Math.log(6.4) / 27.0;
        for (int i = 0; i < mels.length; i++) {
            if (mels[i] < min_log_mel){
                freqs[i] =  f_min + f_sp * mels[i];
            }else{
                freqs[i] = min_log_hz * Math.exp(logstep * (mels[i] - min_log_mel));
            }
        }
        return freqs;
    }

    // Hz to Mel, Slaney
    protected double[] freqToMel(double[] freqs) {
        final double f_min = 0.0;
        final double f_sp = 200.0 / 3;
        double[] mels = new double[freqs.length];
        final double min_log_hz = 1000.0;                         // beginning of log region (Hz)
        final double min_log_mel = (min_log_hz - f_min) / f_sp ;  // # same (Mels)
        final double logstep = Math.log(6.4) / 27.0;              // step size for log region
        for (int i = 0; i < freqs.length; i++) {
            if (freqs[i] < min_log_hz){
                mels[i] = (freqs[i] - f_min) / f_sp;
            }else{
                mels[i] = min_log_mel + Math.log(freqs[i]/min_log_hz) / logstep;
            }
        }
        return mels;
    }

    private double log10(double value) {
        return Math.log(value) / Math.log(10);
    }
}

class MFCCFFT {
    double[] real;
    double[] imag;

    /**
     * Performs Fast Fourier Transformation in place.
     */
    void process(double[] signal) {
        final int numPoints = signal.length;
        // initialize real & imag array
        real = signal;
        imag = new double[numPoints];

        // perform FFT using the real & imag array
        final double pi = Math.PI;
        final int numStages = (int) (Math.log(numPoints) / Math.log(2));
        final int halfNumPoints = numPoints >> 1;
        int j = halfNumPoints;
        // FFT time domain decomposition carried out by "bit reversal sorting"
        // algorithm
        int k;
        for (int i = 1; i < numPoints - 2; i++) {
            if (i < j) {
                // swap
                double tempReal = real[j];
                double tempImag = imag[j];
                real[j] = real[i];
                imag[j] = imag[i];
                real[i] = tempReal;
                imag[i] = tempImag;
            }
            k = halfNumPoints;
            while (k <= j) {
                j -= k;
                k >>= 1;
            }
            j += k;
        }

        // loop for each stage
        for (int stage = 1; stage <= numStages; stage++) {
            int LE = 1;
            for (int i = 0; i < stage; i++) {
                LE <<= 1;
            }
            final int LE2 = LE >> 1;
            double UR = 1;
            double UI = 0;
            // calculate sine & cosine values
            final double SR =  Math.cos(pi / LE2);
            final double SI = -Math.sin(pi / LE2);
            // loop for each sub DFT
            for (int subDFT = 1; subDFT <= LE2; subDFT++) {
                // loop for each butterfly
                for (int butterfly = subDFT - 1; butterfly <= numPoints - 1; butterfly += LE) {
                    int ip = butterfly + LE2;
                    // butterfly calculation
                    double tempReal = (double) (real[ip] * UR - imag[ip] * UI);
                    double tempImag = (double) (real[ip] * UI + imag[ip] * UR);
                    real[ip] = real[butterfly] - tempReal;
                    imag[ip] = imag[butterfly] - tempImag;
                    real[butterfly] += tempReal;
                    imag[butterfly] += tempImag;
                }

                double tempUR = UR;
                UR = tempUR * SR - UI * SI;
                UI = tempUR * SI + UI * SR;
            }
        }
    }
}