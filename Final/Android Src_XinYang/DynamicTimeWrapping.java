package com.neilyxin.www.vibauth;

import android.util.Log;

class DynamicTimeWrapping {
//    private float[][] distanceMatrix;
    private float[] sequenceA;
    private float[] sequenceB;

//    DynamicTimeWrapping() {
//
//    }

    float getDTW(float[] sequenceA, float[] sequenceB) {
        this.sequenceA = sequenceA;
        this.sequenceB = sequenceB;
//        calculateDistanceMatrix(sequenceA, sequenceB);
        return getAccumulatedDistance();
    }

//    private void calculateDistanceMatrix(float[] sequenceA, float[] sequenceB) {
//        if (distanceMatrix==null) {
//            distanceMatrix = new float[sequenceA.length][sequenceB.length];
//        }
//
//        for (int i = 0; i < sequenceA.length; i++) {
//            for (int j = 0; j < sequenceB.length; j++) {
//                distanceMatrix[i][j] = Math.abs(sequenceA[i] - sequenceB[j])*Math.abs(sequenceA[i] - sequenceB[j]);
//            }
//        }
//    }

    private float getDistance(int i, int j) {
        return Math.abs(sequenceA[i] - sequenceB[j])*Math.abs(sequenceA[i] - sequenceB[j]);
    }

    private float getAccumulatedDistance() {
        float accumulatedDistance = getDistance(0,0);
        int i = 0;
        int j = 0;
//        Log.e("TAG", distanceMatrix.length + " " + distanceMatrix[0].length);
        while (i != sequenceA.length - 1 && j!= sequenceA.length - 1) {
            Log.e("TAG", i+" " +j);
            if (getDistance(Math.min(sequenceA.length - 1, i+1), j) >= getDistance(i, Math.min(sequenceA.length - 1, j+1))) {
                if (getDistance(i, Math.min(sequenceA.length - 1, j+1)) >= getDistance(Math.min(sequenceA.length - 1, i+1), Math.min(sequenceA.length - 1, j+1))) {
                    i = i + 1;
                    j = j + 1;
                    accumulatedDistance += getDistance(Math.min(sequenceA.length - 1, i+1), Math.min(sequenceA.length - 1, j+1));
                } else {
                    j = j + 1;
                    accumulatedDistance += getDistance(i, Math.min(sequenceA.length - 1, j+1));
                }
            } else {
                if (getDistance(Math.min(sequenceA.length - 1, i+1),j) >= getDistance(Math.min(sequenceA.length - 1, i+1), Math.min(sequenceA.length - 1, j+1))) {
                    i = i + 1;
                    j = j + 1;
                    accumulatedDistance += getDistance(Math.min(sequenceA.length - 1, i+1), Math.min(sequenceA.length - 1, j+1));
                } else {
                    i = i + 1;
                    accumulatedDistance += getDistance(Math.min(sequenceA.length - 1, i+1), j);
                }
            }
        }
        Log.e("TAG", " " );

        if (i == sequenceA.length - 1 && j == sequenceA.length - 1) {
            Log.e("TAG", i+" " +j);

            return accumulatedDistance;
        } else if(i == sequenceA.length - 1) {
            for (int k = j + 1; k < sequenceA.length; k++) {
                Log.e("TAG", i+" " +k);

                accumulatedDistance += getDistance(i,k);
            }
            return accumulatedDistance;
        } else {
            for (int k = i + 1; k < sequenceA.length; k++) {
                Log.e("TAG", k+" " +j);

                accumulatedDistance += getDistance(k,j);
            }
            return accumulatedDistance;

        }
    }

}
