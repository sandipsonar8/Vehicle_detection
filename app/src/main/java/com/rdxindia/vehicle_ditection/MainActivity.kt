package com.rdxindia.vehicle_ditection

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.rdxindia.vehicle_ditection.Constants.LABELS_PATH
import com.rdxindia.vehicle_ditection.Constants.MODEL_PATH
import com.rdxindia.vehicle_ditection.databinding.ActivityMainBinding
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationManager
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.widget.ImageView
import kotlin.math.max
import kotlin.math.min



class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private var isFrontCamera = false
    private var frameSkipCounter = 0
    private val frameSkipInterval = 2  // Process every 2nd frame

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var detector: Detector

    private lateinit var cameraExecutor: ExecutorService

    // Zoom Variables
    private lateinit var scaleGestureDetector: ScaleGestureDetector
    private var zoomLevel = 1.0f // Default zoom level (1x)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
        detector.setup()

        scaleGestureDetector = ScaleGestureDetector(this, ScaleListener())

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
        getCurrentLocation()
    }

    // ✅ Override onTouchEvent to handle pinch-to-zoom
    override fun onTouchEvent(event: MotionEvent?): Boolean {
        event?.let { scaleGestureDetector.onTouchEvent(it) }
        return true
    }

    // ✅ Zoom Handler Using ScaleGestureDetector
    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            zoomLevel *= detector.scaleFactor
            zoomLevel = max(1.0f, min(zoomLevel, 5.0f)) // Keep zoom between 1x and 5x
            camera?.cameraControl?.setZoomRatio(zoomLevel) // ✅ Apply zoom to camera
            return true
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")
        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val bitmapBuffer = Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                if (isFrontCamera) {
                    postScale(-1f, 1f, imageProxy.width.toFloat(), imageProxy.height.toFloat())
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            detector.detect(rotatedBitmap)
        }

        cameraProvider.unbindAll()
        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector.clear()
        cameraExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() {
        binding.overlay.invalidate()
    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            binding.overlay.apply {
                setResults(boundingBoxes)
                invalidate()
            }

            autoZoom(boundingBoxes)
        }
    }

    // 🚀 Function to adjust zoom based on object size
    private fun autoZoom(boundingBoxes: List<BoundingBox>) {
        if (boundingBoxes.isEmpty() || camera == null) return

        // Get the largest detected object (closest)
        val largestBox = boundingBoxes.maxByOrNull { it.w * it.h } ?: return
        val objectSize = largestBox.w * largestBox.h  // Area of bounding box

        // Define zoom thresholds
        val minZoom = 0.0f   // No zoom
        val maxZoom = 1.0f   // Max zoom
        val zoomFactor: Float

        when {
            objectSize < 0.15f -> zoomFactor = 0.8f  // Object is far, zoom in
            objectSize < 0.3f -> zoomFactor = 0.6f  // Medium distance
            objectSize < 0.5f -> zoomFactor = 0.3f  // Closer, zoom out slightly
            else -> zoomFactor = minZoom           // Object is very close, no zoom
        }

        // Apply zoom smoothly
        camera?.cameraControl?.setLinearZoom(zoomFactor)
    }

    @SuppressLint("MissingPermission")
    private fun getCurrentLocation(): Pair<Double, Double>? {
        // ✅ Check if location permission is granted
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {

            Log.e("Location", "Location permission not granted")
            return null  // Return null if permission is not granted
        }

        val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val providers = locationManager.getProviders(true)

        for (provider in providers.reversed()) {
            val location: Location? = locationManager.getLastKnownLocation(provider)
            if (location != null) {
                return Pair(location.latitude, location.longitude)
            }
        }
        return null  // If no location is available
    }
}


//class MainActivity : AppCompatActivity(), Detector.DetectorListener {
//    private lateinit var binding: ActivityMainBinding
//    private val isFrontCamera = false
//    private var frameSkipCounter = 0
//    private val frameSkipInterval = 2  // Process every 2nd frame
//
//    private var preview: Preview? = null
//    private var imageAnalyzer: ImageAnalysis? = null
//    private var camera: Camera? = null
//    private var cameraProvider: ProcessCameraProvider? = null
//    private lateinit var detector: Detector
//
//    private lateinit var cameraExecutor: ExecutorService
//
//    private lateinit var imageView: ImageView
//    private var scaleFactor = 1.0f
//    private lateinit var scaleGestureDetector: ScaleGestureDetector
//    private val matrix = Matrix()
//
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//        binding = ActivityMainBinding.inflate(layoutInflater)
//        setContentView(binding.root)
//
//        detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
//        detector.setup()
//
//        imageView = findViewById(R.id.imageView)
//        scaleGestureDetector = ScaleGestureDetector(this, ScaleListener())
//
//
//        if (allPermissionsGranted()) {
//            startCamera()
//        } else {
//            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
//        }
//
//        cameraExecutor = Executors.newSingleThreadExecutor()
//        getCurrentLocation();
//    }
//    override fun onTouchEvent(event: MotionEvent?): Boolean {
//        event?.let { scaleGestureDetector.onTouchEvent(it) }
//        return true
//    }
//
//    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
//        override fun onScale(detector: ScaleGestureDetector): Boolean {
//            scaleFactor *= detector.scaleFactor
//            scaleFactor = max(1.0f, min(scaleFactor, 5.0f)) // Limit zoom between 1x and 5x
//            matrix.setScale(scaleFactor, scaleFactor, imageView.width / 2f, imageView.height / 2f)
//            imageView.imageMatrix = matrix
//            return true
//        }
//    }
//
//    private fun startCamera() {
//        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
//        cameraProviderFuture.addListener({
//            cameraProvider  = cameraProviderFuture.get()
//            bindCameraUseCases()
//        }, ContextCompat.getMainExecutor(this))
//    }
//
//    private fun bindCameraUseCases() {
//        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")
//
//        val rotation = binding.viewFinder.display.rotation
//
//        val cameraSelector = CameraSelector
//            .Builder()
//            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
//            .build()
//
//        preview =  Preview.Builder()
//            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
//            .setTargetRotation(rotation)
//            .build()
//
//
//        imageAnalyzer = ImageAnalysis.Builder()
//            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
//            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
//            .setTargetRotation(binding.viewFinder.display.rotation)
//            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
//            .build()
//
//        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
//            val bitmapBuffer =
//                Bitmap.createBitmap(
//                    imageProxy.width,
//                    imageProxy.height,
//                    Bitmap.Config.ARGB_8888
//                )
//            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
//            imageProxy.close()
//
//            val matrix = Matrix().apply {
//                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
//
//                if (isFrontCamera) {
//                    postScale(
//                        -1f,
//                        1f,
//                        imageProxy.width.toFloat(),
//                        imageProxy.height.toFloat()
//                    )
//                }
//            }
//
//            val rotatedBitmap = Bitmap.createBitmap(
//                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
//                matrix, true
//            )
//
//            detector.detect(rotatedBitmap)
//        }
//
//        cameraProvider.unbindAll()
//
//        try {
//            camera = cameraProvider.bindToLifecycle(
//                this,
//                cameraSelector,
//                preview,
//                imageAnalyzer
//            )
//
//            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
//        } catch(exc: Exception) {
//            Log.e(TAG, "Use case binding failed", exc)
//        }
//    }
//
//    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
//        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
//    }
//
//    private val requestPermissionLauncher = registerForActivityResult(
//        ActivityResultContracts.RequestMultiplePermissions()) {
//        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
//    }
//
//    override fun onDestroy() {
//        super.onDestroy()
//        detector.clear()
//        cameraExecutor.shutdown()
//    }
//
//    override fun onResume() {
//        super.onResume()
//        if (allPermissionsGranted()){
//            startCamera()
//        } else {
//            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
//        }
//    }
//
//    companion object {
//        private const val TAG = "Camera"
//        private const val REQUEST_CODE_PERMISSIONS = 10
//        private val REQUIRED_PERMISSIONS = mutableListOf (
//            Manifest.permission.CAMERA
//        ).toTypedArray()
//    }
//
//    override fun onEmptyDetect() {
//        binding.overlay.invalidate()
//    }
//
//    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
//        runOnUiThread {
//            binding.inferenceTime.text = "${inferenceTime}ms"
//            binding.overlay.apply {
//                setResults(boundingBoxes)
//                invalidate()
//            }
//        }
//    }
//    @SuppressLint("MissingPermission")
//    private fun getCurrentLocation(): Pair<Double, Double>? {
//        val locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
//        val providers = locationManager.getProviders(true)
//
//        for (provider in providers.reversed()) {
//            val location: Location? = locationManager.getLastKnownLocation(provider)
//            if (location != null) {
//                return Pair(location.latitude, location.longitude)
//            }
//        }
//        return null  // If location is not available
//    }
//}
