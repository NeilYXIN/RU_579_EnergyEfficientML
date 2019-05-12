package com.neilyxin.www.vibauth;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

class Authenticator {
    private Deque<Integer> predictionLabelQueue;
    private Deque<Float> predictionConfidenceQueue;
    private int size;
    private int validThreshold;
    private Map<Integer, Integer> labelFrequencyMap;
    private int modeLabel;
    private float modeConfidence;
    private int modeFrequency;


    Authenticator(int size, int validThreshold) {
        this.predictionLabelQueue = new LinkedList<>();
        this.predictionConfidenceQueue = new LinkedList<>();
        this.size = size;
        this.validThreshold = validThreshold;
        this.labelFrequencyMap = new HashMap<>();
        this.modeLabel = -1;
        this.modeConfidence = -1;
        this.modeFrequency = -1;
    }

    boolean add(int predictionLabel, float predictionConfidencde) {
        if (predictionLabelQueue.size() == size) {
            predictionLabelQueue.pollFirst();
            predictionConfidenceQueue.pollFirst();
        }
        predictionLabelQueue.addLast(predictionLabel);
        predictionConfidenceQueue.addLast(predictionConfidencde);
        return true;
    }

    Deque<Integer> getPredictionLabelQueue() {
        return predictionLabelQueue;
    }

    Deque<Float> getPredictionConfidenceQueue() {
        return predictionConfidenceQueue;
    }

    void printPredictionQueue() {
        for (int label: predictionLabelQueue) {
            System.out.print(label + " ");
        }
        System.out.print("\n");
        for (float confidence: predictionConfidenceQueue) {
            System.out.print(confidence + " ");
        }
        System.out.print("\n");
    }

    void reset() {
        this.predictionLabelQueue.clear();
        this.predictionConfidenceQueue.clear();
        this.labelFrequencyMap.clear();
        this.modeLabel = -1;
        this.modeConfidence = -1;
        this.modeFrequency = -1;
    }

    /**
     * Set the mode label, frequency, and the confidence, return true if the queue is valid.
     * @return
     */
    boolean getMode() {
        labelFrequencyMap.clear();
        if (predictionLabelQueue.size() != size) {
            return false;
        }
        for (int label: predictionLabelQueue) {
            if (labelFrequencyMap.containsKey(label)) {
                labelFrequencyMap.put(label, labelFrequencyMap.get(label) + 1);
            } else {
                labelFrequencyMap.put(label, 1);
            }
        }
        List<Integer> frequencySet = new ArrayList<>(labelFrequencyMap.values());
        modeFrequency = Collections.max(frequencySet);
        if (modeFrequency < validThreshold) {
            modeFrequency = -1;
            return false;
        } else {
            Iterator<Integer> labelIterator = predictionLabelQueue.iterator();
            Iterator<Float> confidenceIterator = predictionConfidenceQueue.iterator();
            List<Float> modeConfidences = new ArrayList<>();
            while (labelIterator.hasNext() && confidenceIterator.hasNext()) {
                int label = labelIterator.next();
                float confidence = confidenceIterator.next();
                if (labelFrequencyMap.get(label) == modeFrequency) {
                    modeLabel = label;
                    modeConfidences.add(confidence);
                }
            }

            float sumConfidence = 0;
            for (float confidence: modeConfidences) {
                sumConfidence += confidence;
            }
            modeConfidence = sumConfidence/modeFrequency;
            return true;
        }
    }

    int getModeLabel() {
        return modeLabel;
    }

    float getModeConfidence() {
        return modeConfidence;
    }
}
