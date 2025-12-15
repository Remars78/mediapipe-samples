/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.facelandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: FaceLandmarkerResult? = null
    private var paintCursor = Paint()
    private var paintCalibration = Paint()
    private var paintText = Paint()
    private var paintDebug = Paint()

    // --- КОНСТАНТЫ ТОЧЕК ЛИЦА (FACE MESH) ---
    // Левый глаз
    private val LEFT_EYE_INNER = 362
    private val LEFT_EYE_OUTER = 263
    private val LEFT_IRIS = 473
    
    // Правый глаз
    private val RIGHT_EYE_INNER = 33
    private val RIGHT_EYE_OUTER = 133
    private val RIGHT_IRIS = 468

    // --- КАЛИБРОВКА ---
    private enum class CalibrationStage {
        CENTER, TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT, FINISHED
    }
    private var currentStage = CalibrationStage.CENTER
    private var calibrationTimer: Long = 0
    private val CALIBRATION_DELAY_MS = 2000L // 2 сек на точку (быстрее)

    // Храним не координаты, а ОТНОСИТЕЛЬНЫЕ КОЭФФИЦИЕНТЫ (0.0 .. 1.0)
    // Min - взгляд влево/вверх, Max - взгляд вправо/вниз
    private var calibMinX = 1.0f
    private var calibMaxX = -1.0f
    private var calibMinY = 1.0f
    private var calibMaxY = -1.0f

    // --- УМНОЕ СГЛАЖИВАНИЕ (SMOOTHING) ---
    private var cursorX = 0f
    private var cursorY = 0f
    
    // Динамический фильтр:
    // Если глаза двигаются быстро -> alpha высокий (быстрый отклик)
    // Если глаза почти стоят -> alpha низкий (сильное сглаживание дрожания)
    private val MIN_ALPHA = 0.05f // Очень плавно для прицеливания
    private val MAX_ALPHA = 0.6f  // Быстро для рывков

    // --- ЛОГИКА КЛИКА ---
    private var isBlinking = false
    private var blinkStartTime: Long = 0
    private val BLINK_CLICK_DURATION = 1200L // 1.2 сек (быстрее, чтобы не ждать вечность)
    private var isClickState = false

    init {
        paintCursor.color = Color.GREEN
        paintCursor.style = Paint.Style.FILL
        paintCursor.isAntiAlias = true

        paintCalibration.color = Color.RED
        paintCalibration.style = Paint.Style.FILL
        paintCalibration.isAntiAlias = true

        paintText.color = Color.WHITE
        paintText.textSize = 60f
        paintText.isAntiAlias = true
        paintText.setShadowLayer(5f, 0f, 0f, Color.BLACK)
        
        paintDebug.color = Color.YELLOW
        paintDebug.strokeWidth = 2f
        paintDebug.style = Paint.Style.STROKE
    }

    fun setResults(
        faceLandmarkerResults: FaceLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: com.google.mediapipe.tasks.vision.core.RunningMode = com.google.mediapipe.tasks.vision.core.RunningMode.LIVE_STREAM
    ) {
        results = faceLandmarkerResults
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        val result = results ?: return

        if (result.faceLandmarks().isNotEmpty()) {
            val landmarks = result.faceLandmarks()[0]
            
            // --- 1. ВЫЧИСЛЕНИЕ ОТНОСИТЕЛЬНЫХ КООРДИНАТ (МАТЕМАТИКА) ---
            
            // Функция для расчета позиции зрачка внутри глаза (от 0.0 до 1.0)
            fun getEyeRatio(innerIdx: Int, outerIdx: Int, irisIdx: Int): Pair<Float, Float> {
                val inner = landmarks[innerIdx]
                val outer = landmarks[outerIdx]
                val iris = landmarks[irisIdx]

                // Ширина и высота "глазной щели"
                val eyeWidth = outer.x() - inner.x()
                val eyeHeight = outer.y() - inner.y() // Это приближенно, для Y лучше использовать веки
                
                // Проекция зрачка на вектор от внутреннего к внешнему уголку
                // Это делает трекинг устойчивым к повороту головы!
                val irisDistX = iris.x() - inner.x()
                
                // Нормализация X (0.0 = внутренний угол, 1.0 = внешний угол)
                // Для Y используем просто глобальную координату относительно высоты лица,
                // так как веки двигаются при моргании и ломают расчет Y внутри глаза.
                // Но для X этот метод идеален.
                
                val ratioX = irisDistX / eyeWidth
                val ratioY = iris.y() // Для Y пока оставим сырую координату, она стабильнее при моргании
                
                return Pair(ratioX, ratioY)
            }

            // Считаем для обоих глаз и берем среднее
            val leftEye = getEyeRatio(LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_IRIS)
            val rightEye = getEyeRatio(RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_IRIS)

            // Важно: У левого и правого глаза "внутренний угол" с разных сторон.
            // Инвертируем один глаз, чтобы направления совпадали.
            // Левый глаз: Inner(362) -> Outer(263) (Слева направо на экране)
            // Правый глаз: Inner(33) -> Outer(133) (Справа налево, если смотреть зеркально)
            // Усредняем данные:
            val avgRatioX = (leftEye.first + rightEye.first) / 2f
            val avgRawY = (leftEye.second + rightEye.second) / 2f

            // --- 2. ОБРАБОТКА МОРГАНИЯ (BLINK) ---
            val blendshapes = result.faceBlendshapes()
            var blinkScore = 0f
            if (blendshapes.isPresent && blendshapes.get().isNotEmpty()) {
                val shapes = blendshapes.get()[0]
                // Индексы могут варьироваться, но обычно 9 и 10
                if (shapes.size > 10) {
                     val leftBlink = shapes[9].score()
                     val rightBlink = shapes[10].score()
                     blinkScore = (leftBlink + rightBlink) / 2f
                }
            }

            if (blinkScore > 0.6f) {
                if (!isBlinking) {
                    isBlinking = true
                    blinkStartTime = System.currentTimeMillis()
                } else {
                    if (System.currentTimeMillis() - blinkStartTime > BLINK_CLICK_DURATION) {
                        isClickState = true
                    }
                }
            } else {
                isBlinking = false
                isClickState = false
            }

            // --- 3. ЛОГИКА ПРИЛОЖЕНИЯ ---
            if (currentStage != CalibrationStage.FINISHED) {
                processCalibration(canvas, avgRatioX, avgRawY)
            } else {
                processCursor(canvas, avgRatioX, avgRawY)
            }
        }
    }

    private fun processCalibration(canvas: Canvas, eyeRatioX: Float, eyeRawY: Float) {
        val w = width.toFloat()
        val h = height.toFloat()
        val cx = w / 2
        val cy = h / 2
        val padding = 150f

        val now = System.currentTimeMillis()
        if (calibrationTimer == 0L) calibrationTimer = now
        val elapsed = now - calibrationTimer
        val progress = min(1f, elapsed.toFloat() / CALIBRATION_DELAY_MS)
        
        // Рисуем таргет
        var tx = cx; var ty = cy
        var txt = "Центр"

        when (currentStage) {
            CalibrationStage.CENTER -> { tx=cx; ty=cy; txt="Центр" }
            CalibrationStage.TOP_LEFT -> { tx=padding; ty=padding; txt="Верх-Лево" }
            CalibrationStage.TOP_RIGHT -> { tx=w-padding; ty=padding; txt="Верх-Право" }
            CalibrationStage.BOTTOM_RIGHT -> { tx=w-padding; ty=h-padding; txt="Низ-Право" }
            CalibrationStage.BOTTOM_LEFT -> { tx=padding; ty=h-padding; txt="Низ-Лево" }
            else -> {}
        }

        canvas.drawCircle(tx, ty, 40f * progress + 20f, paintCalibration)
        canvas.drawText(txt, cx - 100, cy + 150, paintText)

        // Сбор данных (Ждем 50% времени, чтобы глаз успел доехать, потом пишем)
        if (elapsed > CALIBRATION_DELAY_MS * 0.5) {
            // Инициализация мин/макс первыми значениями
            if (calibMinX > 0.9f) { calibMinX = eyeRatioX; calibMaxX = eyeRatioX; calibMinY = eyeRawY; calibMaxY = eyeRawY }

            // Расширяем диапазон
            calibMinX = min(calibMinX, eyeRatioX)
            calibMaxX = max(calibMaxX, eyeRatioX)
            calibMinY = min(calibMinY, eyeRawY)
            calibMaxY = max(calibMaxY, eyeRawY)
        }

        if (elapsed > CALIBRATION_DELAY_MS) {
            calibrationTimer = 0L
            currentStage = when (currentStage) {
                CalibrationStage.CENTER -> CalibrationStage.TOP_LEFT
                CalibrationStage.TOP_LEFT -> CalibrationStage.TOP_RIGHT
                CalibrationStage.TOP_RIGHT -> CalibrationStage.BOTTOM_RIGHT
                CalibrationStage.BOTTOM_RIGHT -> CalibrationStage.BOTTOM_LEFT
                CalibrationStage.BOTTOM_LEFT -> CalibrationStage.FINISHED
                else -> CalibrationStage.FINISHED
            }
        }
    }

    private fun processCursor(canvas: Canvas, valX: Float, valY: Float) {
        // Защита от схлопывания диапазона
        if (abs(calibMaxX - calibMinX) < 0.01f) calibMaxX += 0.1f
        if (abs(calibMaxY - calibMinY) < 0.01f) calibMaxY += 0.1f

        // 1. Нормализация (Mapping)
        // Для фронтальной камеры инвертируем X (Зеркало)
        var normX = (valX - calibMinX) / (calibMaxX - calibMinX)
        // Для фронталки часто нужно развернуть: 1.0 - normX. 
        // Если курсор бегает инвертировано - уберите "1f - "
        normX = 1f - normX 
        
        var normY = (valY - calibMinY) / (calibMaxY - calibMinY)

        // Кламп с небольшим запасом (чтобы можно было дотянуть до углов)
        normX = max(-0.1f, min(1.1f, normX))
        normY = max(-0.1f, min(1.1f, normY))

        val targetScreenX = normX * width
        val targetScreenY = normY * height

        // 2. ДИНАМИЧЕСКОЕ СГЛАЖИВАНИЕ (Adaptive Filter)
        val dist = Math.hypot((targetScreenX - cursorX).toDouble(), (targetScreenY - cursorY).toDouble()).toFloat()
        
        // Если движение быстрое (>100px) - alpha больше (меньше лаг)
        // Если движение медленное (<20px) - alpha маленькая (жесткая стабилизация)
        var alpha = (dist / 300f).coerceIn(MIN_ALPHA, MAX_ALPHA)
        
        // Линейная интерполяция
        cursorX = cursorX + (targetScreenX - cursorX) * alpha
        cursorY = cursorY + (targetScreenY - cursorY) * alpha

        // 3. Отрисовка
        if (isClickState) {
            paintCursor.color = Color.BLUE
            canvas.drawCircle(cursorX, cursorY, 60f, paintCursor) // Клик больше
            canvas.drawText("CLICK!", cursorX + 60, cursorY, paintText)
        } else {
            paintCursor.color = Color.GREEN
            canvas.drawCircle(cursorX, cursorY, 40f, paintCursor)
            
            // Визуализация подготовки к клику
            if (isBlinking) {
                val duration = System.currentTimeMillis() - blinkStartTime
                val radius = 40f + (duration.toFloat() / BLINK_CLICK_DURATION) * 40f
                paintCursor.style = Paint.Style.STROKE
                paintCursor.strokeWidth = 8f
                canvas.drawCircle(cursorX, cursorY, radius, paintCursor)
                paintCursor.style = Paint.Style.FILL
            }
        }
    }

    fun clear() {
        results = null
        invalidate()
    }
}
