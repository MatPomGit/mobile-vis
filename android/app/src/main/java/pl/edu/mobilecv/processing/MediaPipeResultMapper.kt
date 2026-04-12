package pl.edu.mobilecv.processing

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * Odpowiada za mapowanie wyników MediaPipe na neutralne DTO warstwy domenowej.
 */
class MediaPipeResultMapper {
    /**
     * Mapuje wynik pozy na neutralny model danych.
     */
    fun toPoseDetections(result: PoseLandmarkerResult): PoseDetectionsDto =
        PoseDetectionsDto(result.landmarks().map { it.map(::toLandmarkDto) })

    /**
     * Mapuje wynik dłoni na neutralny model danych.
     */
    fun toHandDetections(result: HandLandmarkerResult): HandDetectionsDto =
        HandDetectionsDto(result.landmarks().map { it.map(::toLandmarkDto) })

    private fun toLandmarkDto(landmark: NormalizedLandmark): LandmarkDto =
        LandmarkDto(x = landmark.x(), y = landmark.y(), z = landmark.z())
}
