package com.ashton.coinclassifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import androidx.exifinterface.media.ExifInterface;
import java.io.IOException;
import java.io.InputStream;

public class ImageUtils {
    public static Bitmap getBitmapFromUri(Context context, Uri uri) throws IOException {
        try (InputStream inputStream = context.getContentResolver().openInputStream(uri)) {
            return BitmapFactory.decodeStream(inputStream);
        }
    }

    public static Bitmap rotateBitmapIfRequired(Bitmap bitmap, Uri uri, Context context) throws IOException {
        try (InputStream input = context.getContentResolver().openInputStream(uri)) {
            ExifInterface exif = new ExifInterface(input);
            int orientation = exif.getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL
            );

            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    return rotateBitmap(bitmap, 90);
                case ExifInterface.ORIENTATION_ROTATE_180:
                    return rotateBitmap(bitmap, 180);
                case ExifInterface.ORIENTATION_ROTATE_270:
                    return rotateBitmap(bitmap, 270);
                default:
                    return bitmap;
            }
        }
    }

    private static Bitmap rotateBitmap(Bitmap bitmap, int degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true
        );
    }
}