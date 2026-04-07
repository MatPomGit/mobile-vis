# Add project specific ProGuard rules here.
# Keep OpenCV classes so native JNI wrappers are not stripped.
-keep class org.opencv.** { *; }

# PyTorch Android Lite keep rules
-keep class org.pytorch.** { *; }
-dontwarn org.pytorch.**

# Keep models related classes if they are used via reflection
-keep class pl.edu.mobilecv.** { *; }

