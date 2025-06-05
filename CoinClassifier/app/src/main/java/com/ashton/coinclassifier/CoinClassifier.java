package com.ashton.coinclassifier;

import android.content.Context;
import android.graphics.Bitmap;
import org.json.JSONException;
import org.json.JSONObject;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import android.util.Log;


import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class CoinClassifier {
    private final Interpreter tflite;
    private final Map<Integer, String> labelMap = new HashMap<>();
    private static final int INPUT_SIZE = 128;

    public CoinClassifier(Context context) throws IOException {
        tflite = new Interpreter(loadModelFile(context, "coin_model.tflite"));
        loadLabelMap(context);
        // Add these imports:
// In constructor after loading model:
        Log.d("TFLite", "Input tensor count: " + tflite.getInputTensorCount());
        for (int i = 0; i < tflite.getInputTensorCount(); i++) {
            Tensor tensor = tflite.getInputTensor(i);
            Log.d("TFLite", String.format(
                    "Input %d: %s (shape: %s)",
                    i, tensor.name(), Arrays.toString(tensor.shape())
            ));
        }
    }

    private ByteBuffer loadModelFile(Context context, String filename) throws IOException {
        InputStream inputStream = context.getAssets().open(filename);
        byte[] bytes = new byte[inputStream.available()];
        int bytesRead = inputStream.read(bytes);
        if (bytesRead != bytes.length) {
            throw new IOException("Failed to read the full model file.");
        }
        inputStream.close();

        ByteBuffer buffer = ByteBuffer.allocateDirect(bytes.length);
        buffer.order(ByteOrder.nativeOrder());
        buffer.put(bytes);
        return buffer;
    }

    private void loadLabelMap(Context context) throws IOException {
        InputStream is = context.getAssets().open("class_indices.json");
        BufferedReader reader = new BufferedReader(new InputStreamReader(is));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            sb.append(line);
        }
        reader.close();

        try {
            JSONObject jsonObject = new JSONObject(sb.toString());
            Iterator<String> keys = jsonObject.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                int index = jsonObject.getInt(key);
                labelMap.put(index, key);
            }
        } catch (JSONException e) {
            throw new IOException("Error parsing class map JSON", e);
        }
    }

    public Result classify(Bitmap bitmap) {
        // Preprocess inputs
        float[][][][] rgbInput = new float[1][INPUT_SIZE][INPUT_SIZE][3];
        float[][][][] grayInput = new float[1][INPUT_SIZE][INPUT_SIZE][1];
        float[][] huInput = new float[1][7];

        preprocessImage(bitmap, rgbInput, grayInput, huInput);

        // Prepare output
        float[][] output = new float[1][labelMap.size()];

        // FIXED INPUT ORDER: Match model's expected sequence
        Object[] inputs = {huInput, rgbInput, grayInput}; // Corrected order
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, output);

        tflite.runForMultipleInputsOutputs(inputs, outputs);

        // Get results (unchanged)
        int maxIdx = 0;
        float maxConfidence = 0;
        for (int i = 0; i < output[0].length; i++) {
            if (output[0][i] > maxConfidence) {
                maxConfidence = output[0][i];
                maxIdx = i;
            }
        }

        return new Result(labelMap.get(maxIdx), maxConfidence);
    }

    private void preprocessImage(
            Bitmap bitmap,
            float[][][][] rgbInput,
            float[][][][] grayInput,
            float[][] huInput
    ) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);
        processRGB(resized, rgbInput[0]);

        Mat grayMat = new Mat();
        Utils.bitmapToMat(resized, grayMat);
        Imgproc.cvtColor(grayMat, grayMat, Imgproc.COLOR_RGBA2GRAY);
        processGray(grayMat, grayInput[0], huInput[0]);
    }

    private void processRGB(Bitmap bitmap, float[][][] output) {
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                int pixel = pixels[i * INPUT_SIZE + j];
                output[i][j][0] = ((pixel >> 16) & 0xFF) / 127.5f - 1.0f;  // R
                output[i][j][1] = ((pixel >> 8) & 0xFF) / 127.5f - 1.0f;   // G
                output[i][j][2] = (pixel & 0xFF) / 127.5f - 1.0f;          // B
            }
        }
    }

    private void processGray(
            Mat gray,
            float[][][] grayOutput,
            float[] huOutput
    ) {
        // 1. Histogram equalization
        Imgproc.equalizeHist(gray, gray);

        // 2. Sharpening kernel
        Mat kernel = new Mat(3, 3, CvType.CV_32F);
        float[] kernelData = {0, -1, 0, -1, 5, -1, 0, -1, 0};
        kernel.put(0, 0, kernelData);
        Imgproc.filter2D(gray, gray, -1, kernel);

        // 3. CLAHE contrast enhancement
        org.opencv.imgproc.CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
        clahe.apply(gray, gray);

        // 4. Normalization
        Core.normalize(gray, gray, 0, 255, Core.NORM_MINMAX);

        // 5. Gaussian blur
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        // 6. Canny edge detection
        Mat edges = new Mat();
        Imgproc.Canny(gray, edges, 100, 200);

        // 7. Contour detection and drawing
        Mat contourImg = Mat.zeros(edges.size(), CvType.CV_8UC1);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(
                edges, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE
        );

        if (!contours.isEmpty()) {
            Imgproc.drawContours(contourImg, contours, -1, new Scalar(255), 1);
        }

        // Prepare grayscale output
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                grayOutput[i][j][0] = (float)(contourImg.get(i, j)[0] / 255.0f);
            }
        }

        // Compute Hu Moments
        computeHuMoments(contourImg, huOutput);

        // Release Mats
        edges.release();
        contourImg.release();
        hierarchy.release();
        for (MatOfPoint contour : contours) {
            contour.release();
        }
    }

    private void computeHuMoments(Mat contourImg, float[] output) {
        org.opencv.imgproc.Moments moments = Imgproc.moments(contourImg);
        Mat huMoments = new Mat();
        Imgproc.HuMoments(moments, huMoments);

        for (int i = 0; i < 7; i++) {
            double hu = huMoments.get(i, 0)[0];
            double sign = Math.signum(hu);
            output[i] = (float)(-sign * Math.log10(Math.abs(hu) + 1e-10));
        }
        huMoments.release();
    }

    public void close() {
        tflite.close();
    }

    public static class Result {
        public final String className;
        public final float confidence;

        public Result(String className, float confidence) {
            this.className = className;
            this.confidence = confidence;
        }
    }
}