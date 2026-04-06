package pl.edu.mobilecv

import org.opencv.core.*
import org.opencv.imgproc.Imgproc

/**
 * Utility class for basic OpenCV image filters.
 */
object LegacyFilters {

    fun applyGrayscale(src: Mat): Mat {
        val res = Mat()
        Imgproc.cvtColor(src, res, Imgproc.COLOR_RGBA2GRAY)
        val out = Mat()
        Imgproc.cvtColor(res, out, Imgproc.COLOR_GRAY2RGBA)
        res.release()
        return out
    }

    fun applyCanny(src: Mat): Mat {
        val gray = Mat(); val blurred = Mat(); val edges = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.GaussianBlur(gray, blurred, Size(5.0, 5.0), 0.0)
        Imgproc.Canny(blurred, edges, 50.0, 150.0)
        Imgproc.cvtColor(edges, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); blurred.release(); edges.release()
        return res
    }

    fun applyGaussianBlur(src: Mat): Mat {
        val res = Mat()
        Imgproc.GaussianBlur(src, res, Size(5.0, 5.0), 0.0)
        return res
    }

    fun applyCartoon(src: Mat): Mat {
        val rgb = Mat(); val smoothed = Mat()
        Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        Imgproc.bilateralFilter(rgb, smoothed, 9, 75.0, 75.0)
        val gray = Mat(); val blurred = Mat(); val edges = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.medianBlur(gray, blurred, 7)
        Imgproc.Canny(blurred, edges, 80.0, 200.0)
        val edgesInv = Mat(); Core.bitwise_not(edges, edgesInv)
        val dilatedEdges = Mat()
        Imgproc.dilate(edgesInv, dilatedEdges, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0)))
        val edgeMask3ch = Mat()
        Imgproc.cvtColor(dilatedEdges, edgeMask3ch, Imgproc.COLOR_GRAY2RGB)
        val maskedSmoothed = Mat()
        Core.bitwise_and(smoothed, edgeMask3ch, maskedSmoothed)
        val res = Mat()
        Imgproc.cvtColor(maskedSmoothed, res, Imgproc.COLOR_RGB2RGBA)
        rgb.release(); smoothed.release(); gray.release(); blurred.release()
        edges.release(); edgesInv.release(); dilatedEdges.release()
        edgeMask3ch.release(); maskedSmoothed.release()
        return res
    }

    fun applySepia(src: Mat): Mat {
        val rgb = Mat(); Imgproc.cvtColor(src, rgb, Imgproc.COLOR_RGBA2RGB)
        val rgb32f = Mat(); rgb.convertTo(rgb32f, CvType.CV_32FC3)
        val kernel = Mat(3, 3, CvType.CV_32F)
        kernel.put(0, 0, floatArrayOf(0.393f, 0.769f, 0.189f))
        kernel.put(1, 0, floatArrayOf(0.349f, 0.686f, 0.168f))
        kernel.put(2, 0, floatArrayOf(0.272f, 0.534f, 0.131f))
        val sepia32f = Mat(); Core.transform(rgb32f, sepia32f, kernel)
        val sepia8u = Mat(); sepia32f.convertTo(sepia8u, CvType.CV_8UC3)
        val res = Mat(); Imgproc.cvtColor(sepia8u, res, Imgproc.COLOR_RGB2RGBA)
        rgb.release(); rgb32f.release(); kernel.release(); sepia32f.release(); sepia8u.release()
        return res
    }

    fun applyThreshold(src: Mat): Mat {
        val gray = Mat(); val thresh = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.threshold(gray, thresh, 127.0, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.cvtColor(thresh, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); thresh.release()
        return res
    }

    fun applySobel(src: Mat): Mat {
        val gray = Mat(); val sx = Mat(); val sy = Mat(); val ax = Mat(); val ay = Mat(); val c = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Sobel(gray, sx, CvType.CV_16S, 1, 0)
        Imgproc.Sobel(gray, sy, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(sx, ax); Core.convertScaleAbs(sy, ay)
        Core.addWeighted(ax, 0.5, ay, 0.5, 0.0, c)
        Imgproc.cvtColor(c, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); sx.release(); sy.release(); ax.release(); ay.release(); c.release()
        return res
    }

    fun applyLaplacian(src: Mat): Mat {
        val gray = Mat(); val lap = Mat(); val abs = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Laplacian(gray, lap, CvType.CV_16S)
        Core.convertScaleAbs(lap, abs)
        Imgproc.cvtColor(abs, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); lap.release(); abs.release()
        return res
    }

    fun applyMorphology(src: Mat, op: Int, kernelSize: Int): Mat {
        val res = Mat()
        val k = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size((2 * kernelSize + 1).toDouble(), (2 * kernelSize + 1).toDouble()))
        if (op == -1) { // Custom for Dilate
            Imgproc.dilate(src, res, k)
        } else if (op == -2) { // Custom for Erode
            Imgproc.erode(src, res, k)
        } else {
            Imgproc.morphologyEx(src, res, op, k)
        }
        k.release()
        return res
    }

    fun applyMedianBlur(src: Mat): Mat {
        val res = Mat()
        Imgproc.medianBlur(src, res, 5)
        return res
    }

    fun applyBilateralFilter(src: Mat): Mat {
        val res = Mat(); val gray = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2RGB)
        Imgproc.bilateralFilter(gray, res, 9, 75.0, 75.0)
        val out = Mat(); Imgproc.cvtColor(res, out, Imgproc.COLOR_RGB2RGBA)
        gray.release(); res.release()
        return out
    }

    fun applyBoxFilter(src: Mat): Mat {
        val res = Mat()
        Imgproc.boxFilter(src, res, -1, Size(5.0, 5.0))
        return res
    }

    fun applyAdaptiveThreshold(src: Mat): Mat {
        val gray = Mat(); val thresh = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.adaptiveThreshold(gray, thresh, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2.0)
        Imgproc.cvtColor(thresh, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); thresh.release()
        return res
    }

    fun applyHistogramEqualization(src: Mat): Mat {
        val gray = Mat(); val equ = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.equalizeHist(gray, equ)
        Imgproc.cvtColor(equ, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); equ.release()
        return res
    }

    fun applyScharr(src: Mat): Mat {
        val gray = Mat(); val sx = Mat(); val sy = Mat(); val ax = Mat(); val ay = Mat(); val c = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        Imgproc.Scharr(gray, sx, CvType.CV_16S, 1, 0)
        Imgproc.Scharr(gray, sy, CvType.CV_16S, 0, 1)
        Core.convertScaleAbs(sx, ax); Core.convertScaleAbs(sy, ay)
        Core.addWeighted(ax, 0.5, ay, 0.5, 0.0, c)
        Imgproc.cvtColor(c, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); sx.release(); sy.release(); ax.release(); ay.release(); c.release()
        return res
    }

    fun applyPrewitt(src: Mat): Mat {
        val gray = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val kernelX = Mat(3, 3, CvType.CV_32F)
        kernelX.put(0, 0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0)
        val kernelY = Mat(3, 3, CvType.CV_32F)
        kernelY.put(0, 0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        val gradX = Mat(); val gradY = Mat()
        Imgproc.filter2D(gray, gradX, -1, kernelX)
        Imgproc.filter2D(gray, gradY, -1, kernelY)
        val absX = Mat(); val absY = Mat()
        Core.convertScaleAbs(gradX, absX); Core.convertScaleAbs(gradY, absY)
        val combined = Mat()
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, combined)
        Imgproc.cvtColor(combined, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); kernelX.release(); kernelY.release(); gradX.release(); gradY.release(); absX.release(); absY.release(); combined.release()
        return res
    }

    fun applyRoberts(src: Mat): Mat {
        val gray = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val kernelX = Mat(2, 2, CvType.CV_32F)
        kernelX.put(0, 0, 1.0, 0.0, 0.0, -1.0)
        val kernelY = Mat(2, 2, CvType.CV_32F)
        kernelY.put(0, 0, 0.0, 1.0, -1.0, 0.0)
        val gradX = Mat(); val gradY = Mat()
        Imgproc.filter2D(gray, gradX, -1, kernelX)
        Imgproc.filter2D(gray, gradY, -1, kernelY)
        val absX = Mat(); val absY = Mat()
        Core.convertScaleAbs(gradX, absX); Core.convertScaleAbs(gradY, absY)
        val combined = Mat()
        Core.addWeighted(absX, 0.5, absY, 0.5, 0.0, combined)
        Imgproc.cvtColor(combined, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); kernelX.release(); kernelY.release(); gradX.release(); gradY.release(); absX.release(); absY.release(); combined.release()
        return res
    }

    fun applyInvert(src: Mat): Mat {
        val res = Mat()
        Core.bitwise_not(src, res)
        return res
    }

    fun applyEmboss(src: Mat): Mat {
        val gray = Mat(); val embossed = Mat(); val res = Mat()
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_RGBA2GRAY)
        val kernel = Mat(3, 3, CvType.CV_32F)
        kernel.put(0, 0, -2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0)
        Imgproc.filter2D(gray, embossed, -1, kernel)
        Core.add(embossed, Scalar(128.0), embossed)
        Imgproc.cvtColor(embossed, res, Imgproc.COLOR_GRAY2RGBA)
        gray.release(); embossed.release(); kernel.release()
        return res
    }

    fun applyPixelate(src: Mat, blockSize: Int): Mat {
        val small = Mat()
        val res = Mat()
        Imgproc.resize(src, small, Size(src.cols().toDouble() / blockSize, src.rows().toDouble() / blockSize), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.resize(small, res, Size(src.cols().toDouble(), src.rows().toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST)
        small.release()
        return res
    }
}
