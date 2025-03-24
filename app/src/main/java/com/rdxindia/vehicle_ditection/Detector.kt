package com.rdxindia.vehicle_ditection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.*
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener
) {

    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options().apply {
            setUseNNAPI(true)
            setNumThreads(4)
        }
        interpreter = Interpreter(model, options)

        interpreter?.getInputTensor(0)?.shape()?.let {
            tensorWidth = it[1]
            tensorHeight = it[2]
        }
        interpreter?.getOutputTensor(0)?.shape()?.let {
            numChannel = it[1]
            numElements = it[2]
        }

        try {
            context.assets.open(labelPath).bufferedReader().useLines { lines ->
                labels.addAll(lines.filter { it.isNotEmpty() })
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun clear() {
        interpreter?.close()
        interpreter = null
    }

    fun detect(frame: Bitmap) {
        interpreter ?: return
        if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) return

        val startTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)
        val tensorImage = TensorImage(DataType.FLOAT32).apply { load(resizedBitmap) }
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)
        interpreter?.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)
        val inferenceTime = SystemClock.uptimeMillis() - startTime

        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        detectorListener.onDetect(bestBoxes, inferenceTime)

        for (box in bestBoxes) {
            if (box.clsName in listOf("truck", "bus", "van")) {
                val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())

                captureAndSaveImage(frame, timestamp, false)

                Handler(Looper.getMainLooper()).postDelayed({
                    val zoomedFrame = getZoomedBitmap(frame, box)
                    captureAndSaveImage(zoomedFrame, timestamp, true)
                }, 4000)

                break
            }
        }
    }

    private fun getZoomedBitmap(original: Bitmap, box: BoundingBox): Bitmap {
        val width = original.width
        val height = original.height

        val x1 = (box.x1 * width).toInt().coerceIn(0, width - 1)
        val y1 = (box.y1 * height).toInt().coerceIn(0, height - 1)
        val x2 = (box.x2 * width).toInt().coerceIn(0, width - 1)
        val y2 = (box.y2 * height).toInt().coerceIn(0, height - 1)

        val cropWidth = (x2 - x1).coerceAtLeast(1)
        val cropHeight = (y2 - y1).coerceAtLeast(1)

        val croppedBitmap = Bitmap.createBitmap(original, x1, y1, cropWidth, cropHeight)

        return upscaleImage(croppedBitmap, width, height)
    }

    private fun upscaleImage(bitmap: Bitmap, newWidth: Int, newHeight: Int): Bitmap {
        val matrix = Matrix().apply {
            postScale(
                newWidth.toFloat() / bitmap.width,
                newHeight.toFloat() / bitmap.height
            )
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun captureAndSaveImage(bitmap: Bitmap, timestamp: String, isZoomed: Boolean) {
        val storageDir = File(
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES),
            "VehicleDetections"
        )
        if (!storageDir.exists()) storageDir.mkdirs()

        val fileName = if (isZoomed) "IMG_${timestamp}_zoom.jpg" else "IMG_${timestamp}.jpg"
        val file = File(storageDir, fileName)

        FileOutputStream(file).use { fos ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)
            fos.flush()
        }

        Log.d("Capture", "Image saved: ${file.absolutePath}")

        Handler(Looper.getMainLooper()).post {
            val msg = if (isZoomed) "Zoomed image saved ✅" else "Original image saved ✅"
            Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
        }
    }

    private fun applyNMS(boxes: List<BoundingBox>): List<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }


    private fun bestBox(array: FloatArray): List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            for (j in 4 until numChannel) {
                val arrayIdx = c + numElements * j
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                val clsName = labels[maxIdx]
                val cx = array[c]
                val cy = array[c + numElements]
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w / 2F)
                val y1 = cy - (h / 2F)
                val x2 = cx + (w / 2F)
                val y2 = cy + (h / 2F)

                if (x1 in 0F..1F && y1 in 0F..1F && x2 in 0F..1F && y2 in 0F..1F) {
                    boundingBoxes.add(
                        BoundingBox(x1, y1, x2, y2, cx, cy, w, h, maxConf, maxIdx, clsName)
                    )
                }
            }
        }

        return if (boundingBoxes.isEmpty()) null else applyNMS(boundingBoxes)
    }



    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.5F
        private const val IOU_THRESHOLD = 0.4F
    }
}
