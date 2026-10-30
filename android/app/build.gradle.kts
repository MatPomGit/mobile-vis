plugins {
    alias(libs.plugins.android.application)
}
val releaseKeystore = System.getenv("MOBILE_VIS_KEYSTORE_PATH")
val releaseStorePassword = System.getenv("MOBILE_VIS_STORE_PASSWORD")
val releaseKeyAlias = System.getenv("MOBILE_VIS_KEY_ALIAS")
val releaseKeyPassword = System.getenv("MOBILE_VIS_KEY_PASSWORD")
val releaseSigningConfigured = listOf(
    releaseKeystore, releaseStorePassword, releaseKeyAlias, releaseKeyPassword
).all { !it.isNullOrBlank() }

android {
    namespace = "pl.edu.mobilecv"
    compileSdk = 36

    defaultConfig {
        applicationId = "pl.edu.mobilecv"
        minSdk = 24
        targetSdk = 36
        versionName = rootProject.extra["app_version_name"].toString()
        versionCode = rootProject.extra["app_version_code"] as Int

        ndk {
            abiFilters.addAll(listOf("armeabi-v7a", "arm64-v8a"))
        }
    }

    signingConfigs {
        if (releaseSigningConfigured) {
            create("release") {
                storeFile = file(requireNotNull(releaseKeystore))
                storePassword = requireNotNull(releaseStorePassword)
                keyAlias = requireNotNull(releaseKeyAlias)
                keyPassword = requireNotNull(releaseKeyPassword)
            }
        }
    }

    buildTypes {
        release {
            if (releaseSigningConfigured) {
                signingConfig = signingConfigs.getByName("release")
            }
            // Enable minification, obfuscation, and optimization
            isMinifyEnabled = true
            // Enable resource shrinking
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
        getByName("debug") {
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    bundle {
        abi {
            enableSplit = true
        }
        density {
            enableSplit = true
        }
        language {
            enableSplit = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        viewBinding = true
    }
    packaging {
        jniLibs {
            pickFirsts += "**/libc++_shared.so"
        }
    }
    buildToolsVersion = "36.0.0"
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.datastore.preferences)
    implementation(libs.androidx.work.runtime.ktx)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.video)
    implementation(libs.androidx.camera.view)
    implementation(libs.opencv)
    implementation(libs.okhttp)
    implementation(libs.mediapipe.tasks.vision)
    implementation(libs.pytorch.android)
    implementation(libs.tflite.main)
    implementation(libs.tflite.gpu)
    implementation(libs.tflite.gpuapi)
    implementation(libs.tflite.support)
    testImplementation(libs.junit)
}
