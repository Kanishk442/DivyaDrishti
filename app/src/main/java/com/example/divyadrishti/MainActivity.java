package com.example.divyadrishti;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.RectF;
import android.media.Image;
import android.os.Build;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.task.vision.detector.Detection;
import org.tensorflow.lite.task.vision.detector.ObjectDetector;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@ExperimentalGetImage
public class MainActivity extends AppCompatActivity {
    private static final int CAMERA_PERMISSION_CODE = 101;
    private static final long COOLDOWN_MS = 1500;
    private static final float REFERENCE_OBJECT_HEIGHT_PX = 600f;
    private static final float REFERENCE_DISTANCE_METERS = 1.0f;

    private PreviewView previewView;
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private TextToSpeech textToSpeech;
    private Vibrator vibrator;
    private ObjectDetector objectDetector;
    private EnhancedYoloDetector yoloDetector;

    private Button btnAutoMode, btnTapMode;
    private TextView tvDetectionResult, tvTapInstruction, tvTitle, tvSubtitle, tvInstructions;
    private long lastAnnounceTime = 0;
    private long lastVibrationTime = 0;
    private boolean detectionStarted = false;
    private String lastDescription = "";
    private String currentMode = ""; // "auto" or "tap"
    private boolean useYolo = true;

    private static final Set<String> IMPORTANT_OBJECTS = new HashSet<String>() {{
        add("person"); add("car"); add("bicycle"); add("motorcycle");
        add("bus"); add("truck"); add("traffic light"); add("stop sign");
        add("stairs"); add("door"); add("chair"); add("table");
        add("book"); add("cell phone"); add("bottle"); add("cup");
        add("sign"); add("text"); add("menu"); add("card");
    }};

    private ExecutorService analysisExecutor;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initializeViews();
        setupTTS();
        setupVibrator();
        setupClickListeners();
        loadDetector();

        analysisExecutor = Executors.newSingleThreadExecutor();
    }

    private void initializeViews() {
        btnAutoMode = findViewById(R.id.btnAutoMode);
        btnTapMode = findViewById(R.id.btnTapMode);
        previewView = findViewById(R.id.cameraPreview);
        tvDetectionResult = findViewById(R.id.tvDetectionResult);
        tvTapInstruction = findViewById(R.id.tvTapInstruction);
        tvTitle = findViewById(R.id.tvTitle);
        tvSubtitle = findViewById(R.id.tvSubtitle);
        tvInstructions = findViewById(R.id.tvInstructions);
    }

    private void setupTTS() {
        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.US);
                speakDescription("Divya Drishti started. Please choose a mode.");
            }
        });
    }

    private void setupVibrator() {
        vibrator = (Vibrator) getSystemService(VIBRATOR_SERVICE);
    }

    private void setupClickListeners() {
        btnAutoMode.setOnClickListener(v -> startMode("auto"));
        btnTapMode.setOnClickListener(v -> startMode("tap"));

        previewView.setOnClickListener(v -> {
            if (currentMode.equals("tap") && detectionStarted) {
                speakDescription("Processing");
                vibrateAlert(100);
                processSingleDetection();
            }
        });
    }

    private void startMode(String mode) {
        currentMode = mode;
        btnAutoMode.setVisibility(View.GONE);
        btnTapMode.setVisibility(View.GONE);
        tvTitle.setVisibility(View.GONE);
        tvSubtitle.setVisibility(View.GONE);
        tvInstructions.setVisibility(View.GONE);

        previewView.setVisibility(View.VISIBLE);
        tvDetectionResult.setVisibility(View.VISIBLE);

        if (mode.equals("tap")) {
            tvTapInstruction.setVisibility(View.VISIBLE);
            speakDescription("Tap to detect mode. Tap anywhere on screen to identify objects.");
        } else {
            speakDescription("Auto detection mode started. Point your camera to detect objects.");
        }

        detectionStarted = true;
        checkCameraPermission();
    }

    private void processSingleDetection() {
        // Capture current frame and process it
        previewView.post(() -> {
            Bitmap bitmap = previewView.getBitmap();
            if (bitmap != null) {
                try {
                    List<DetectionResult> results;
                    if (useYolo && yoloDetector != null) {
                        results = yoloDetector.detect(bitmap);
                    } else if (objectDetector != null) {
                        TensorImage tensorImage = TensorImage.fromBitmap(bitmap);
                        List<Detection> detections = objectDetector.detect(tensorImage);
                        results = convertToDetectionResults(detections);
                    } else {
                        results = new ArrayList<>();
                    }
                    handleDetectionResults(results);
                } catch (Exception e) {
                    Log.e("TapDetection", "Error in tap detection", e);
                }
            }
        });
    }

    private void loadDetector() {
        new Thread(() -> {
            try {
                if (useYolo) {
                    // Try to load YOLO detector
                    yoloDetector = new EnhancedYoloDetector(
                            getApplicationContext(),
                            "yolov8n_float16.tflite",
                            "coco-labels-2014_2017.txt"
                    );
                    Log.i("Detector", "YOLO detector loaded successfully");
                } else {
                    // Fall back to original model
                    loadEfficientDetModel();
                }
            } catch (Exception e) {
                Log.e("Detector", "Failed to load YOLO detector, falling back", e);
                loadEfficientDetModel();
            }
        }).start();
    }

    private void loadEfficientDetModel() {
        try {
            ObjectDetector.ObjectDetectorOptions options = ObjectDetector.ObjectDetectorOptions.builder()
                    .setMaxResults(5)
                    .setScoreThreshold(0.6f)
                    .build();

            objectDetector = ObjectDetector.createFromFileAndOptions(
                    this, "efficientdet_lite0.tflite", options
            );
        } catch (IOException e) {
            Log.e("TFLite", "Failed to load EfficientDet model", e);
        }
    }

    private List<DetectionResult> convertToDetectionResults(List<Detection> detections) {
        List<DetectionResult> results = new ArrayList<>();
        for (Detection detection : detections) {
            if (detection.getCategories() != null && !detection.getCategories().isEmpty()) {
                results.add(new DetectionResult(
                        detection.getCategories().get(0).getLabel(),
                        detection.getCategories().get(0).getScore(),
                        detection.getBoundingBox()
                ));
            }
        }
        return results;
    }

    private void checkCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(
                    this,
                    new String[]{Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_CODE
            );
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show();
            finish();
        }
    }

    private void startCamera() {
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                Preview preview = new Preview.Builder().build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());

                // Only set up continuous analysis for auto mode
                if (currentMode.equals("auto")) {
                    ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build();

                    imageAnalysis.setAnalyzer(analysisExecutor, imageProxy -> {
                        if (detectionStarted) processImageForObjects(imageProxy);
                        else imageProxy.close();
                    });

                    cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis);
                } else {
                    // For tap mode, just bind preview
                    cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview);
                }
            } catch (Exception e) {
                Log.e("CameraX", "Error setting up camera", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void processImageForObjects(ImageProxy imageProxy) {
        Image mediaImage = imageProxy.getImage();
        if (mediaImage == null) {
            imageProxy.close();
            return;
        }

        try {
            Bitmap bitmap = yuvImageToBitmap(mediaImage, imageProxy.getImageInfo().getRotationDegrees());
            if (bitmap == null) {
                imageProxy.close();
                return;
            }

            List<DetectionResult> results;
            if (useYolo && yoloDetector != null) {
                results = yoloDetector.detect(bitmap);
            } else if (objectDetector != null) {
                TensorImage tensorImage = TensorImage.fromBitmap(bitmap);
                List<Detection> detections = objectDetector.detect(tensorImage);
                results = convertToDetectionResults(detections);
            } else {
                results = new ArrayList<>();
            }

            handleDetectionResults(results);
        } catch (Exception e) {
            Log.e("Detection", "Error during detection", e);
        } finally {
            imageProxy.close();
        }
    }

    private void handleDetectionResults(List<DetectionResult> results) {
        long now = System.currentTimeMillis();
        StringBuilder speechDescription = new StringBuilder();
        StringBuilder resultText = new StringBuilder();

        for (DetectionResult result : results) {
            String label = result.getLabel();
            float confidence = result.getConfidence();
            RectF bbox = result.getBoundingBox();
            float distance = estimateDistance(bbox.height());

            String display = String.format("%s (%.0f%%)", label, confidence * 100);
            if (distance > 0) {
                display += String.format(" - %.1fm", distance);
            }
            resultText.append(display).append("\n");

            if (confidence > 0.6f && IMPORTANT_OBJECTS.contains(label.toLowerCase(Locale.US))) {
                if (distance > 0) {
                    speechDescription.append(label).append(" at ").append(String.format("%.1f", distance)).append(" meters. ");
                } else {
                    speechDescription.append(label).append(". ");
                }
            }
        }

        String newDescription = speechDescription.toString().trim();
        String displayText = resultText.toString().trim();

        runOnUiThread(() -> tvDetectionResult.setText(displayText));

        if (!newDescription.isEmpty() && (!newDescription.equals(lastDescription) || now - lastAnnounceTime > COOLDOWN_MS)) {
            lastAnnounceTime = now;
            lastDescription = newDescription;
            speakDescription(newDescription);
            vibrateAlert(newDescription);
        }
    }

    private float estimateDistance(float heightPx) {
        if (heightPx <= 0) return -1;
        float ratio = REFERENCE_OBJECT_HEIGHT_PX / heightPx;
        float approxDistance = ratio * REFERENCE_DISTANCE_METERS;
        return Math.max(0.2f, Math.min(approxDistance, 10f));
    }

    private void speakDescription(String desc) {
        if (textToSpeech != null && desc != null && !desc.isEmpty()) {
            textToSpeech.stop();
            textToSpeech.speak(desc, TextToSpeech.QUEUE_FLUSH, null, "object_detection");
        }
    }

    private void vibrateAlert(String speechDesc) {
        if (vibrator == null || !vibrator.hasVibrator()) return;

        long now = System.currentTimeMillis();
        if (now - lastVibrationTime < 1000) return;
        lastVibrationTime = now;

        long vibrationDuration = 100;
        if (speechDesc != null) {
            String lowerDesc = speechDesc.toLowerCase(Locale.US);
            if (lowerDesc.contains("person") || lowerDesc.contains("car") || lowerDesc.contains("bicycle")) {
                vibrationDuration = 200;
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(vibrationDuration, VibrationEffect.DEFAULT_AMPLITUDE));
        } else {
            vibrator.vibrate(vibrationDuration);
        }
    }

    private void vibrateAlert(int duration) {
        if (vibrator != null && vibrator.hasVibrator()) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator.vibrate(VibrationEffect.createOneShot(duration, VibrationEffect.DEFAULT_AMPLITUDE));
            } else {
                vibrator.vibrate(duration);
            }
        }
    }

    private Bitmap yuvImageToBitmap(Image image, int rotationDegrees) {
        if (image.getFormat() != ImageFormat.YUV_420_888) return null;

        YuvToRgbConverter converter = new YuvToRgbConverter(this);
        Bitmap bitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
        converter.yuvToRgb(image, bitmap);

        if (rotationDegrees != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotationDegrees);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        }

        return bitmap;
    }

    private static class YuvToRgbConverter {
        private final Context context;

        YuvToRgbConverter(Context context) {
            this.context = context;
        }

        void yuvToRgb(Image image, Bitmap outputBitmap) {
            Image.Plane[] planes = image.getPlanes();
            ByteBuffer yBuffer = planes[0].getBuffer();
            ByteBuffer uBuffer = planes[1].getBuffer();
            ByteBuffer vBuffer = planes[2].getBuffer();

            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();

            byte[] yData = new byte[ySize];
            byte[] uData = new byte[uSize];
            byte[] vData = new byte[vSize];

            yBuffer.get(yData, 0, ySize);
            uBuffer.get(uData, 0, uSize);
            vBuffer.get(vData, 0, vSize);

            int width = image.getWidth();
            int height = image.getHeight();
            int[] pixels = new int[width * height];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int yValue = yData[y * width + x] & 0xFF;
                    int uIndex = (y / 2) * (width / 2) + (x / 2);
                    int vIndex = (y / 2) * (width / 2) + (x / 2);

                    if (uIndex >= uData.length) uIndex = uData.length - 1;
                    if (vIndex >= vData.length) vIndex = vData.length - 1;


                    int uValue = uData[uIndex] & 0xFF;
                    int vValue = vData[vIndex] & 0xFF;

                    int r = (int) (yValue + 1.402 * (vValue - 128));
                    int g = (int) (yValue - 0.344 * (uValue - 128) - 0.714 * (vValue - 128));
                    int b = (int) (yValue + 1.772 * (uValue - 128));

                    r = Math.max(0, Math.min(255, r));
                    g = Math.max(0, Math.min(255, g));
                    b = Math.max(0, Math.min(255, b));

                    pixels[y * width + x] = 0xff000000 | (r << 16) | (g << 8) | b;
                }
            }

            outputBitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (detectionStarted) {
            detectionStarted = false;
            if (cameraProviderFuture != null && cameraProviderFuture.isDone()) {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    cameraProvider.unbindAll();
                } catch (Exception e) {
                    Log.e("CameraLifecycle", "Error cleaning up camera", e);
                }
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (previewView.getVisibility() == View.VISIBLE && !detectionStarted) {
            detectionStarted = true;
            startCamera();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (yoloDetector != null) {
            yoloDetector.close();
        }
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        if (analysisExecutor != null && !analysisExecutor.isShutdown()) {
            analysisExecutor.shutdownNow();
        }
        if (objectDetector != null) {
            objectDetector.close();
        }

        if (cameraProviderFuture != null && cameraProviderFuture.isDone()) {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                cameraProvider.unbindAll();
            } catch (Exception e) {
                Log.e("CameraLifecycle", "Error cleaning up camera", e);
            }
        }
    }

    // EnhancedYoloDetector inner class - FIXED VERSION
    public class EnhancedYoloDetector {
        private static final String TAG = "EnhancedYoloDetector";
        private static final float MINIMUM_CONFIDENCE = 0.5f;
        private static final float MINIMUM_PERSON_CONFIDENCE = 0.65f;
        private static final float NMS_THRESHOLD = 0.6f;

        private final Interpreter interpreter;
        private final List<String> labels;
        private final int inputSize;
        private final ImageProcessor imageProcessor;

        public EnhancedYoloDetector(Context context, String modelFile, String labelFile) throws IOException {
            // Load model with basic options
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);

            // Load model file
            MappedByteBuffer modelBuffer = FileUtil.loadMappedFile(context, modelFile);
            interpreter = new Interpreter(modelBuffer, options);

            // Get input shape
            int[] inputShape = interpreter.getInputTensor(0).shape();
            inputSize = inputShape[1];

            // Create image processor
            imageProcessor = new ImageProcessor.Builder()
                    .add(new ResizeOp(inputSize, inputSize, ResizeOp.ResizeMethod.BILINEAR))
                    .add(new NormalizeOp(0f, 255f))
                    .build();

            // Load labels manually
            labels = loadLabelsManually(context, labelFile);

            Log.i(TAG, "YOLO detector initialized with input size: " + inputSize);

            // Debug model info
            debugModelInfo();
        }

        // Manual label loading method
        private List<String> loadLabelsManually(Context context, String labelFile) {
            List<String> labels = new ArrayList<>();
            try {
                InputStream inputStream = context.getAssets().open(labelFile);
                BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
                String line;
                while ((line = reader.readLine()) != null) {
                    labels.add(line.trim());
                }
                reader.close();
                Log.i(TAG, "Loaded " + labels.size() + " labels from " + labelFile);
            } catch (IOException e) {
                Log.e(TAG, "Failed to load labels: " + e.getMessage());
            }
            return labels;
        }

        // Debug method to show model information
        private void debugModelInfo() {
            try {
                Log.d(TAG, "=== YOLO Model Debug Info ===");

                // Input tensor info
                int inputTensorCount = interpreter.getInputTensorCount();
                Log.d(TAG, "Input tensors: " + inputTensorCount);

                for (int i = 0; i < inputTensorCount; i++) {
                    int[] inputShape = interpreter.getInputTensor(i).shape();
                    Log.d(TAG, "Input " + i + " shape: " + java.util.Arrays.toString(inputShape));
                    Log.d(TAG, "Input " + i + " data type: " + interpreter.getInputTensor(i).dataType());
                }

                // Output tensor info
                int outputTensorCount = interpreter.getOutputTensorCount();
                Log.d(TAG, "Output tensors: " + outputTensorCount);

                for (int i = 0; i < outputTensorCount; i++) {
                    int[] outputShape = interpreter.getOutputTensor(i).shape();
                    Log.d(TAG, "Output " + i + " shape: " + java.util.Arrays.toString(outputShape));
                    Log.d(TAG, "Output " + i + " data type: " + interpreter.getOutputTensor(i).dataType());
                }

                Log.d(TAG, "Labels count: " + labels.size());
                if (labels.size() > 0) {
                    Log.d(TAG, "First few labels: " + labels.subList(0, Math.min(10, labels.size())));
                }

            } catch (Exception e) {
                Log.e(TAG, "Error debugging model: " + e.getMessage());
            }
        }

        public List<DetectionResult> detect(Bitmap bitmap) {
            try {
                Log.d(TAG, "Starting detection on bitmap: " + bitmap.getWidth() + "x" + bitmap.getHeight());

                // Convert bitmap to TensorImage
                TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                tensorImage.load(bitmap);
                tensorImage = imageProcessor.process(tensorImage);

                // Get input buffer
                ByteBuffer inputBuffer = tensorImage.getBuffer();

                // Prepare output based on ACTUAL model output shape [1, 84, 8400]
                int[] outputShape = interpreter.getOutputTensor(0).shape();
                Log.d(TAG, "Output shape: " + java.util.Arrays.toString(outputShape));

                // Create appropriate 3D output array
                float[][][] output = new float[outputShape[0]][outputShape[1]][outputShape[2]];

                // Run inference
                long startTime = System.currentTimeMillis();
                interpreter.run(inputBuffer, output);
                long inferenceTime = System.currentTimeMillis() - startTime;

                Log.d(TAG, "Inference completed in " + inferenceTime + "ms");

                // Process results - YOLOv8 output is [1, 84, 8400]
                return processYolov8Output(output[0], bitmap.getWidth(), bitmap.getHeight());
            } catch (Exception e) {
                Log.e(TAG, "Error during detection: " + e.getMessage(), e);
                return new ArrayList<>();
            }
        }

        private List<DetectionResult> processYolov8Output(float[][] output, int originalWidth, int originalHeight) {
            List<DetectionResult> detections = new ArrayList<>();

            // YOLOv8 output shape: [84, 8400]
            // Where: 84 = 4 (bbox coordinates) + 80 (class probabilities for COCO)
            //        8400 = number of predictions (grid cells)

            int numClasses = 80; // COCO has 80 classes
            int numPredictions = output[0].length; // Should be 8400

            Log.d(TAG, "Processing " + numPredictions + " predictions with " + numClasses + " classes");

            for (int i = 0; i < numPredictions; i++) {
                // Extract bounding box coordinates (cx, cy, w, h) - these are normalized [0,1]
                float cx = output[0][i]; // center x
                float cy = output[1][i]; // center y
                float w = output[2][i];  // width
                float h = output[3][i];  // height

                // Find class with maximum probability
                int classId = -1;
                float maxConfidence = 0;
                for (int j = 0; j < numClasses; j++) {
                    float confidence = output[4 + j][i];
                    if (confidence > maxConfidence) {
                        maxConfidence = confidence;
                        classId = j;
                    }
                }

                // Apply confidence threshold
                if (maxConfidence < MINIMUM_CONFIDENCE) continue;

                // Convert class ID to label (COCO classes are 0-79)
                String label = "object";
                if (classId >= 0 && classId < labels.size()) {
                    label = labels.get(classId);
                } else {
                    Log.w(TAG, "Invalid class ID: " + classId);
                    continue;
                }

                // Apply special handling for person detection
                if ("person".equals(label)) {
                    if (maxConfidence < MINIMUM_PERSON_CONFIDENCE) continue;

                    // Additional checks to reduce false positives
                    if (!isLikelyRealPerson(cx, cy, w, h, originalWidth, originalHeight, maxConfidence)) {
                        Log.d(TAG, "Filtered out false person detection with confidence: " + maxConfidence);
                        continue;
                    }
                }

                // Convert normalized coordinates to pixel coordinates
                float left = (cx - w / 2) * originalWidth;
                float top = (cy - h / 2) * originalHeight;
                float right = (cx + w / 2) * originalWidth;
                float bottom = (cy + h / 2) * originalHeight;

                // Ensure coordinates are within image bounds
                left = Math.max(0, left);
                top = Math.max(0, top);
                right = Math.min(originalWidth, right);
                bottom = Math.min(originalHeight, bottom);

                // Skip invalid boxes
                if (right <= left || bottom <= top) {
                    Log.d(TAG, "Skipping invalid box: " + left + "," + top + "," + right + "," + bottom);
                    continue;
                }

                RectF boundingBox = new RectF(left, top, right, bottom);
                detections.add(new DetectionResult(label, maxConfidence, boundingBox));

                Log.d(TAG, "Detected: " + label + " with confidence: " + maxConfidence);
            }

            Log.d(TAG, "Found " + detections.size() + " detections before NMS");

            // Apply Non-Maximum Suppression to remove duplicate detections
            return nms(detections);
        }

        private boolean isLikelyRealPerson(float cx, float cy, float w, float h, int imageWidth, int imageHeight, float confidence) {
            // Convert to pixel dimensions
            float width = w * imageWidth;
            float height = h * imageHeight;

            // Check aspect ratio - people are typically taller than wide
            float aspectRatio = height / width;
            if (aspectRatio < 1.2f) {
                Log.d(TAG, "Filtered person: aspect ratio too wide: " + aspectRatio);
                return false;
            }

            // Check size - very small detections are more likely to be false positives
            float boxArea = width * height;
            float imageArea = imageWidth * imageHeight;
            float areaRatio = boxArea / imageArea;

            if (areaRatio < 0.01f) {
                Log.d(TAG, "Filtered person: too small: " + areaRatio);
                return false;
            }

            return true;
        }

        private List<DetectionResult> nms(List<DetectionResult> detections) {
            List<DetectionResult> filteredDetections = new ArrayList<>();

            // Sort by confidence descending
            detections.sort((a, b) -> Float.compare(b.getConfidence(), a.getConfidence()));

            while (!detections.isEmpty()) {
                // Get the detection with highest confidence
                DetectionResult current = detections.get(0);
                filteredDetections.add(current);
                detections.remove(0);

                // Remove overlapping detections
                List<DetectionResult> toRemove = new ArrayList<>();
                for (DetectionResult detection : detections) {
                    if (current.getLabel().equals(detection.getLabel())) {
                        float iou = calculateIoU(current.getBoundingBox(), detection.getBoundingBox());
                        if (iou > NMS_THRESHOLD) {
                            toRemove.add(detection);
                        }
                    }
                }
                detections.removeAll(toRemove);
            }

            Log.d(TAG, "Found " + filteredDetections.size() + " detections after NMS");
            return filteredDetections;
        }

        private float calculateIoU(RectF box1, RectF box2) {
            float intersectionLeft = Math.max(box1.left, box2.left);
            float intersectionTop = Math.max(box1.top, box2.top);
            float intersectionRight = Math.min(box1.right, box2.right);
            float intersectionBottom = Math.min(box1.bottom, box2.bottom);

            if (intersectionRight < intersectionLeft || intersectionBottom < intersectionTop) {
                return 0f;
            }

            float intersectionArea = (intersectionRight - intersectionLeft) * (intersectionBottom - intersectionTop);
            float area1 = (box1.right - box1.left) * (box1.bottom - box1.top);
            float area2 = (box2.right - box2.left) * (box2.bottom - box2.top);
            float unionArea = area1 + area2 - intersectionArea;

            return intersectionArea / unionArea;
        }

        public void close() {
            if (interpreter != null) {
                interpreter.close();
            }
        }
    }

    // DetectionResult inner class
    public class DetectionResult {
        private final String label;
        private final float confidence;
        private final RectF boundingBox;

        public DetectionResult(String label, float confidence, RectF boundingBox) {
            this.label = label;
            this.confidence = confidence;
            this.boundingBox = boundingBox;
        }

        public String getLabel() {
            return label;
        }

        public float getConfidence() {
            return confidence;
        }

        public RectF getBoundingBox() {
            return boundingBox;
        }

        @Override
        public String toString() {
            return String.format("%s (%.2f) [%.1f, %.1f, %.1f, %.1f]",
                    label, confidence,
                    boundingBox.left, boundingBox.top, boundingBox.right, boundingBox.bottom);
        }
    }
}