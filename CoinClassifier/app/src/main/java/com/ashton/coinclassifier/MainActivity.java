package com.ashton.coinclassifier;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "CoinClassifier";
    private ImageView imagePreview;
    private TextView resultText;
    private TextView confidenceText;
    private CoinClassifier classifier;
    private String currentPhotoPath;
    private Bitmap currentBitmap;

    // Permission launchers
    private final ActivityResultLauncher<String> requestCameraPermission =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) openCamera();
                else showToast("Camera permission required");
            });

    private final ActivityResultLauncher<String> requestStoragePermission =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) openGallery();
                else showToast("Storage permission required");
            });

    // Activity launchers
    private final ActivityResultLauncher<Intent> takePictureLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == RESULT_OK) {
                    handleCameraResult();
                } else {
                    showToast("Picture not taken");
                }
            });

    private final ActivityResultLauncher<String> pickImageLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) handleGalleryResult(uri);
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI elements
        imagePreview = findViewById(R.id.imagePreview);
        resultText = findViewById(R.id.resultText);
        confidenceText = findViewById(R.id.confidenceText);

        // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed");
            showToast("OpenCV initialization failed");
            finish();
        }

        // Initialize classifier
        try {
            classifier = new CoinClassifier(this);
            resultText.setText("Classifier ready");
        } catch (Exception e) {
            resultText.setText("Error: " + e.getMessage());
            Log.e(TAG, "Classifier initialization failed", e);
        }

        // Setup button listeners
        findViewById(R.id.captureButton).setOnClickListener(v -> checkCameraPermission());
        findViewById(R.id.loadButton).setOnClickListener(v -> checkStoragePermission());
        findViewById(R.id.classifyButton).setOnClickListener(v -> classifyImage());
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA);
        }
    }

    private void checkStoragePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            // Android 13+ uses new media permission
            if (ContextCompat.checkSelfPermission(
                    this, Manifest.permission.READ_MEDIA_IMAGES) == PackageManager.PERMISSION_GRANTED) {
                openGallery();
            } else {
                requestStoragePermission.launch(Manifest.permission.READ_MEDIA_IMAGES);
            }
        } else {
            // Older versions use READ_EXTERNAL_STORAGE
            if (ContextCompat.checkSelfPermission(
                    this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                openGallery();
            } else {
                requestStoragePermission.launch(Manifest.permission.READ_EXTERNAL_STORAGE);
            }
        }
    }

    private void openCamera() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) == null) {
            showToast("No camera app found");
            return;
        }

        File photoFile = createImageFile();
        if (photoFile == null) {
            showToast("Error creating image file");
            return;
        }

        currentPhotoPath = photoFile.getAbsolutePath();
        Uri photoURI = FileProvider.getUriForFile(this,
                getPackageName() + ".fileprovider",
                photoFile);

        // Grant temporary permissions
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
        takePictureIntent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION);

        takePictureLauncher.launch(takePictureIntent);
    }

    private File createImageFile() {
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);

        try {
            return File.createTempFile(
                    imageFileName,
                    ".jpg",
                    storageDir
            );
        } catch (IOException e) {
            Log.e(TAG, "Error creating image file", e);
            return null;
        }
    }

    private void openGallery() {
        pickImageLauncher.launch("image/*");
    }

    private void handleCameraResult() {
        try {
            // Load and rotate the image
            currentBitmap = loadAndRotateImage(currentPhotoPath);
            imagePreview.setImageBitmap(currentBitmap);
            resultText.setText("Image captured");
        } catch (IOException | OutOfMemoryError e) {
            showToast("Error loading image");
            Log.e(TAG, "Image load error", e);
        }
    }

    private void handleGalleryResult(Uri uri) {
        try {
            // Load scaled and rotated image
            currentBitmap = loadScaledBitmap(uri, 1024, 1024);
            currentBitmap = rotateBitmapIfRequired(currentBitmap, uri);
            imagePreview.setImageBitmap(currentBitmap);
            resultText.setText("Image loaded");
        } catch (IOException | OutOfMemoryError e) {
            showToast("Error loading image");
            Log.e(TAG, "Gallery image load error", e);
        }
    }

    private Bitmap loadAndRotateImage(String photoPath) throws IOException {
        // First load with proper orientation
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(photoPath, options);

        options.inSampleSize = calculateInSampleSize(options, 1024, 1024);
        options.inJustDecodeBounds = false;

        Bitmap bitmap = BitmapFactory.decodeFile(photoPath, options);

        // Apply rotation
        ExifInterface exif = new ExifInterface(photoPath);
        return rotateBitmap(bitmap, exif);
    }

    private Bitmap loadScaledBitmap(Uri uri, int maxWidth, int maxHeight) throws IOException {
        InputStream input = getContentResolver().openInputStream(uri);
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(input, null, options);
        if (input != null) input.close();

        options.inSampleSize = calculateInSampleSize(options, maxWidth, maxHeight);
        options.inJustDecodeBounds = false;

        input = getContentResolver().openInputStream(uri);
        Bitmap bitmap = BitmapFactory.decodeStream(input, null, options);
        if (input != null) input.close();

        return bitmap;
    }

    private int calculateInSampleSize(BitmapFactory.Options options, int reqWidth, int reqHeight) {
        final int height = options.outHeight;
        final int width = options.outWidth;
        int inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {
            final int halfHeight = height / 2;
            final int halfWidth = width / 2;

            while ((halfHeight / inSampleSize) >= reqHeight
                    && (halfWidth / inSampleSize) >= reqWidth) {
                inSampleSize *= 2;
            }
        }
        return inSampleSize;
    }

    private Bitmap rotateBitmapIfRequired(Bitmap bitmap, Uri uri) throws IOException {
        InputStream input = getContentResolver().openInputStream(uri);
        ExifInterface exif = new ExifInterface(input);
        if (input != null) input.close();
        return rotateBitmap(bitmap, exif);
    }

    private Bitmap rotateBitmap(Bitmap bitmap, ExifInterface exif) {
        int orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_UNDEFINED
        );

        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.postRotate(90);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.postRotate(180);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.postRotate(270);
                break;
            default:
                return bitmap;
        }

        return Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true
        );
    }

    private void classifyImage() {
        if (currentBitmap == null) {
            showToast("No image to classify");
            return;
        }

        resultText.setText("Classifying...");
        confidenceText.setText("Processing...");

        new Thread(() -> {
            try {
                CoinClassifier.Result result = classifier.classify(currentBitmap);
                runOnUiThread(() -> {
                    resultText.setText("Prediction: " + result.className);
                    confidenceText.setText(String.format("Confidence: %.1f%%", result.confidence * 100));
                });
            } catch (Exception e) {
                Log.e(TAG, "Classification error", e);
                runOnUiThread(() -> {
                    resultText.setText("Classification failed");
                    confidenceText.setText("Error: " + e.getMessage());
                });
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        if (classifier != null) {
            classifier.close();
        }
        super.onDestroy();
    }

    private void showToast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show();
    }
}