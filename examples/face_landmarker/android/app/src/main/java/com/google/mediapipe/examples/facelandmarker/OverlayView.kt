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
import kotlin.math.max
import kotlin.math.min

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: FaceLandmarkerResult? = null
    private var paintCursor = Paint()
    private var paintCalibration = Paint()
    private var paintText = Paint()

    // --- Eye Tracking Logic Variables ---
    // Индексы зрачков в Face Mesh (468 - левый, 473 - правый)
    private val LEFT_IRIS_CENTER = 468
    private val RIGHT_IRIS_CENTER = 473

    // Калибровка
    private enum class CalibrationStage {
        CENTER, TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT, FINISHED
    }
    private var currentStage = CalibrationStage.CENTER
    private var calibrationTimer: Long = 0
    private val CALIBRATION_DELAY_MS = 3000L // Время на каждую точку (3 сек)

    // Границы движения глаз (будут определены при калибровке)
    private var eyeMinX = Float.MAX_VALUE
    private var eyeMaxX = Float.MIN_VALUE
    private var eyeMinY = Float.MAX_VALUE
    private var eyeMaxY = Float.MIN_VALUE

    // Сглаживание курсора (Linear Interpolation)
    private var cursorX = 0f
    private var cursorY = 0f
    private val SMOOTHING_FACTOR = 0.2f // Чем меньше, тем плавнее, но больше задержка

    // Логика клика (моргание)
    private var isBlinking = false
    private var blinkStartTime: Long = 0
    private val BLINK_CLICK_DURATION = 2000L // 2 секунды
    private var isClickState = false

    init {
        // Настройка кисти курсора
        paintCursor.color = Color.GREEN
        paintCursor.style = Paint.Style.FILL
        paintCursor.isAntiAlias = true

        // Настройка кисти калибровки
        paintCalibration.color = Color.RED
        paintCalibration.style = Paint.Style.FILL
        paintCalibration.isAntiAlias = true

        // Настройка текста
        paintText.color = Color.WHITE
        paintText.textSize = 50f
        paintText.isAntiAlias = true
        paintText.setShadowLayer(5f, 0f, 0f, Color.BLACK)
    }

    fun setResults(
        faceLandmarkerResults: FaceLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: com.google.mediapipe.tasks.vision.core.RunningMode = com.google.mediapipe.tasks.vision.core.RunningMode.LIVE_STREAM
    ) {
        results = faceLandmarkerResults
        // Перерисовываем экран при каждом новом результате
        invalidate()
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        val result = results ?: return

        if (result.faceLandmarks().isNotEmpty()) {
            val landmarks = result.faceLandmarks()[0]
            val blendshapes = result.faceBlendshapes()

            // 1. Получаем координаты зрачков (нормализованные 0.0 - 1.0)
            val leftIris = landmarks[LEFT_IRIS_CENTER]
            val rightIris = landmarks[RIGHT_IRIS_CENTER]

            // Средняя точка взгляда (сырая)
            val rawEyeX = (leftIris.x() + rightIris.x()) / 2f
            val rawEyeY = (leftIris.y() + rightIris.y()) / 2f

            // 2. Логика моргания (Click)
            // Индексы blendshapes могут отличаться, обычно ищем по имени категории
            // MediaPipe blendshapes: eyeBlinkLeft, eyeBlinkRight
            var blinkScore = 0f
            if (blendshapes.isPresent && blendshapes.get().isNotEmpty()) {
                val shapes = blendshapes.get()[0]
                // Ищем blendshapes для моргания. Индексы 9 и 10 обычно eyeBlinkLeft/Right,
                // но лучше пройтись по списку, если он именован, или использовать hardcode если порядок фиксирован.
                // В стандартной модели MediaPipe Face Mesh V2:
                // eyeBlinkLeft ~ index 9
                // eyeBlinkRight ~ index 10
                // Для надежности просто берем значения, если они доступны по индексу
                if (shapes.size > 10) {
                     val leftBlink = shapes[9].score()
                     val rightBlink = shapes[10].score()
                     blinkScore = (leftBlink + rightBlink) / 2f
                }
            }

            if (blinkScore > 0.5f) { // Глаза закрыты
                if (!isBlinking) {
                    isBlinking = true
                    blinkStartTime = System.currentTimeMillis()
                } else {
                    val duration = System.currentTimeMillis() - blinkStartTime
                    if (duration > BLINK_CLICK_DURATION) {
                        isClickState = true // Активируем клик
                    }
                }
            } else {
                isBlinking = false
                isClickState = false // Сброс клика
            }

            // 3. Машина состояний (Калибровка vs Работа)
            if (currentStage != CalibrationStage.FINISHED) {
                runCalibration(canvas, rawEyeX, rawEyeY)
            } else {
                runMouseControl(canvas, rawEyeX, rawEyeY)
            }
        }
    }

    private fun runCalibration(canvas: Canvas, eyeX: Float, eyeY: Float) {
        val w = width.toFloat()
        val h = height.toFloat()
        val cx = w / 2
        val cy = h / 2
        val padding = 100f

        var targetX = 0f
        var targetY = 0f
        var instruction = ""

        // Логика переключения этапов по таймеру
        val now = System.currentTimeMillis()
        if (calibrationTimer == 0L) calibrationTimer = now
        
        // Рисуем таймер/прогресс
        val elapsed = now - calibrationTimer
        val progress = min(1f, elapsed.toFloat() / CALIBRATION_DELAY_MS)
        val radius = 30f + (20f * progress) // Пульсация

        // Если прошло время, сохраняем данные и идем дальше
        if (elapsed > CALIBRATION_DELAY_MS) {
            // Захват крайних точек
            eyeMinX = min(eyeMinX, eyeX)
            eyeMaxX = max(eyeMaxX, eyeX)
            eyeMinY = min(eyeMinY, eyeY)
            eyeMaxY = max(eyeMaxY, eyeY)

            // Переход к следующему этапу
            calibrationTimer = 0L
            currentStage = when (currentStage) {
                CalibrationStage.CENTER -> CalibrationStage.TOP_LEFT
                CalibrationStage.TOP_LEFT -> CalibrationStage.TOP_RIGHT
                CalibrationStage.TOP_RIGHT -> CalibrationStage.BOTTOM_RIGHT
                CalibrationStage.BOTTOM_RIGHT -> CalibrationStage.BOTTOM_LEFT
                CalibrationStage.BOTTOM_LEFT -> CalibrationStage.FINISHED
                else -> CalibrationStage.FINISHED
            }
            if (currentStage == CalibrationStage.FINISHED) return // Сразу переходим к управлению
        }

        // Определение координат цели для текущего этапа
        when (currentStage) {
            CalibrationStage.CENTER -> { targetX = cx; targetY = cy; instruction = "Смотрите в ЦЕНТР" }
            CalibrationStage.TOP_LEFT -> { targetX = padding; targetY = padding; instruction = "Смотрите ВЛЕВО-ВВЕРХ" }
            CalibrationStage.TOP_RIGHT -> { targetX = w - padding; targetY = padding; instruction = "Смотрите ВПРАВО-ВВЕРХ" }
            CalibrationStage.BOTTOM_RIGHT -> { targetX = w - padding; targetY = h - padding; instruction = "Смотрите ВПРАВО-ВНИЗ" }
            CalibrationStage.BOTTOM_LEFT -> { targetX = padding; targetY = h - padding; instruction = "Смотрите ВЛЕВО-ВНИЗ" }
            else -> {}
        }

        // Рисуем цель
        paintCalibration.color = Color.RED
        canvas.drawCircle(targetX, targetY, radius, paintCalibration)
        canvas.drawText(instruction, cx - 200, cy + 100, paintText)
    }

    private fun runMouseControl(canvas: Canvas, eyeX: Float, eyeY: Float) {
        // Защита от деления на ноль, если калибровка прошла плохо
        if (eyeMaxX == eyeMinX) eyeMaxX += 0.01f
        if (eyeMaxY == eyeMinY) eyeMaxY += 0.01f

        // 1. Нормализация (приведение координат глаз 0..1 относительно калибровки)
        // Инвертируем X, так как камера зеркалит (если это фронталка)
        // Для фронтальной камеры движение зрачка влево (на изображении) это движение вправо на экране
        var normX = (eyeX - eyeMinX) / (eyeMaxX - eyeMinX)
        var normY = (eyeY - eyeMinY) / (eyeMaxY - eyeMinY)

        // Клэмп (чтобы не вылетал за границы)
        normX = max(0f, min(1f, normX))
        normY = max(0f, min(1f, normY))

        // 2. Преобразование в координаты экрана
        // Инверсия X нужна, если FaceLandmarker не настроен на зеркалирование.
        // Обычно для селфи-мыши: взгляд влево -> курсор влево.
        // Проверьте на устройстве: возможно, понадобится `1f - normX`.
        val targetScreenX = (1f - normX) * width 
        val targetScreenY = normY * height

        // 3. Сглаживание (Lerp)
        cursorX = cursorX + (targetScreenX - cursorX) * SMOOTHING_FACTOR
        cursorY = cursorY + (targetScreenY - cursorY) * SMOOTHING_FACTOR

        // 4. Отрисовка курсора
        if (isClickState) {
            paintCursor.color = Color.BLUE // Клик активен
            canvas.drawText("CLICK!", cursorX + 40, cursorY, paintText)
        } else {
            paintCursor.color = Color.GREEN // Просто движение
        }

        canvas.drawCircle(cursorX, cursorY, 30f, paintCursor)
        
        // Индикация прогресса клика (если глаза закрыты, но еще не клик)
        if (isBlinking && !isClickState) {
             val duration = System.currentTimeMillis() - blinkStartTime
             val sweep = (duration.toFloat() / BLINK_CLICK_DURATION) * 360f
             paintCursor.style = Paint.Style.STROKE
             paintCursor.strokeWidth = 5f
             canvas.drawArc(cursorX - 40, cursorY - 40, cursorX + 40, cursorY + 40, -90f, sweep, false, paintCursor)
             paintCursor.style = Paint.Style.FILL // Возвращаем заливку
        }
    }

    fun clear() {
        results = null
        paintCursor.reset()
        invalidate()
    }
}
