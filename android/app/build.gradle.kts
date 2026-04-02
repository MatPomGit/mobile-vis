plugins {
    alias(libs.plugins.android.application)
}

android {
    signingConfigs {
        create("release") {
            val keystorePath: String? =
                findProperty("ANDROID_KEYSTORE_PATH") as String?
                    ?: System.getenv("ANDROID_KEYSTORE_PATH")
            val keystorePassword: String? =
                findProperty("ANDROID_KEYSTORE_PASSWORD") as String?
                    ?: System.getenv("ANDROID_KEYSTORE_PASSWORD")
            val keyAlias: String? =
                findProperty("ANDROID_KEY_ALIAS") as String?
                    ?: System.getenv("ANDROID_KEY_ALIAS")
            val keyPassword: String? =
                findProperty("ANDROID_KEY_PASSWORD") as String?
                    ?: System.getenv("ANDROID_KEY_PASSWORD")

            if (!keystorePath.isNullOrBlank()) {
                val keystoreFile = file(keystorePath)
                if (!keystoreFile.exists()) {
                    throw GradleException(
                        "Android release signing: keystore file not found at '$keystorePath'. " +
                            "Set ANDROID_KEYSTORE_PATH (or the Gradle property) to a valid path."
                    )
                }
                if (keystorePassword.isNullOrBlank() || keyAlias.isNullOrBlank() || keyPassword.isNullOrBlank()) {
                    throw GradleException(
                        "Android release signing: ANDROID_KEYSTORE_PATH is set but one or more " +
                            "required credentials are missing " +
                            "(ANDROID_KEYSTORE_PASSWORD, ANDROID_KEY_ALIAS, ANDROID_KEY_PASSWORD)."
                    )
                }
                storeFile = keystoreFile
                storePassword = keystorePassword
                this.keyAlias = keyAlias
                this.keyPassword = keyPassword
            }
        }
    }
    namespace = "pl.edu.mobilecv"
    compileSdk = 36

    defaultConfig {
        applicationId = "pl.edu.mobilecv"
        minSdk = 24
        targetSdk = 36
        versionName = rootProject.extra["app_version_code"].toString()
        versionCode = (rootProject.extra["app_version_code"] as Number).toInt()
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("release")
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
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
    flavorDimensions += listOf("wymiar_A")
    productFlavors {
        create("flavor_A1") {
            dimension = "wymiar_A"
        }
    }
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
