package com.neilyxin.www.vibauth;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.Queue;

class FileHelper {

    private Queue<BufferedWriter> bufferedWriters;
    private Queue<FileWriter> fileWriters;

    FileHelper() {
        this.bufferedWriters = new LinkedList<>();
        this.fileWriters = new LinkedList<>();
    }

    BufferedWriter getBufferWriter(String path) {
        File file = new File(path);
        if (!file.exists()) {
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
                return null;
            }
        }
        try {
            FileWriter fileWriter = new FileWriter(file, false);
            fileWriters.add(fileWriter);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            bufferedWriters.add(bufferedWriter);
            return bufferedWriter;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }

    void closeAllWriters() {
        while (!bufferedWriters.isEmpty()) {
            BufferedWriter bufferedWriter = bufferedWriters.poll();
            try {
                bufferedWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        while (!fileWriters.isEmpty()) {
            FileWriter fileWriter = fileWriters.poll();
            try {
                fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

}
