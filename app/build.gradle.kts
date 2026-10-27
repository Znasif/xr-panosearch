plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.znasif.xrpanosearch"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.znasif.xrpanosearch"
        minSdk = 34
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        ndk { abiFilters += "arm64-v8a" }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = true
    }
    packaging {
        jniLibs {
            // Extract .so files to nativeLibraryDir as real files: the CDSP loads
            // libQnnHtpV79Skel.so via ADSP_LIBRARY_PATH and cannot read it from inside the APK.
            useLegacyPackaging = true
            // qnn-runtime ships every backend and HTP arch; Aura (SXR2330P) needs only HTP V79.
            // libQnnHtpPrepare.so (90 MB) is for on-device graph compilation, which the AOT
            // context binaries inside the .pte files don't need.
            excludes += listOf(
                "**/libQnnDsp*.so", "**/libQnnGpu.so", "**/libQnnHtpPrepare.so",
                "**/libQnnHtpV6*.so", "**/libQnnHtpV7[0-5]*.so", "**/libQnnHtpV81*.so",
            )
        }
    }
    lint {
        disable += "RestrictedApi"
    }
}

dependencies {
    // QNN runtime for the in-process HTP backend (libQnnHtp.so, V79 stub + skel, libQnnSystem.so).
    // Must match the QAIRT version the .pte files and libqnn_executorch_backend.so were built with.
    implementation("com.qualcomm.qti:qnn-runtime:2.46.0")

    val composeBom = platform("androidx.compose:compose-bom:2025.02.00")
    implementation(composeBom)
    implementation("androidx.activity:activity-compose:1.10.0")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.runtime:runtime")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    implementation("androidx.xr.compose:compose:1.0.0-beta01")
    implementation("androidx.xr.runtime:runtime:1.0.0-rc02")
    implementation("androidx.xr.scenecore:scenecore:1.0.0-rc02")
    implementation("androidx.xr.arcore:arcore:1.0.0-rc02")
}
