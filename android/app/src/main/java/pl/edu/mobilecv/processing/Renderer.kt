package pl.edu.mobilecv.processing

import android.graphics.Bitmap

/**
 * Warstwa renderowania odpowiedzialna za finalne nakładki na klatkę.
 */
class Renderer(
    private val fpsOverlayDrawer: (Bitmap) -> Unit,
) {
    /**
     * Zwraca klatkę po opcjonalnym dorysowaniu elementów HUD.
     */
    fun render(bitmap: Bitmap, showFpsOverlay: Boolean): Bitmap {
        if (showFpsOverlay) {
            fpsOverlayDrawer(bitmap)
        }
        return bitmap
    }
}
