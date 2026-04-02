plugins {
    alias(libs.plugins.android.application)
}
val storePass by extra(123456)
val keyAlias by extra("key0")

android {
    namespace = "pl.edu.mobilecv"
    compileSdk = 36

    defaultConfig {
        applicationId = "pl.edu.mobilecv"
        minSdk = 24
        targetSdk = 36
        versionName = rootProject.extra["app_version_code"].toString()
        versionCode = (rootProject.extra["app_version_code"] as Number).toInt()
    }

    signingConfigs {
        create("release") {
            // Replace with your actual keystore information
            storeFile = file("C:\\Users\\matpo\\.android\\keystore.jks")
            storePassword = "13456"
            keyAlias = "key0"
            keyPassword = "123456"
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
            // Disable minification for release
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
        getByName("debug") {
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        viewBinding = true
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
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.video)
    implementation(libs.androidx.camera.view)
    implementation(libs.opencv)
    implementation(libs.okhttp)
    implementation(libs.mediapipe.tasks.vision)
}
