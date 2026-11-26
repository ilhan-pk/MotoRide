plugins {
    id("com.android.application")
}

android {
    namespace = "com.ilhan.motoride"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.ilhan.motoride"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64") }
    }

    buildTypes {
        release { isMinifyEnabled = false }
    }
}

val jniDir = file("src/main/jniLibs")

val hasCargo: Boolean by lazy {
    try {
        ProcessBuilder("cargo", "--version").redirectErrorStream(true).start().waitFor() == 0
    } catch (e: Exception) {
        false
    }
}

// Builds app/rust into jniLibs/<abi>/libmotoride.so with cargo-ndk.
// Skipped automatically when cargo is not installed (e.g. building on a phone).
val cargoNdkBuild by tasks.registering(Exec::class) {
    onlyIf { hasCargo }
    workingDir = file("rust")
    val cmd = mutableListOf("cargo", "ndk")
    listOf("arm64-v8a", "armeabi-v7a", "x86_64").forEach { cmd += listOf("-t", it) }
    cmd += listOf("-o", jniDir.absolutePath, "build", "--release", "--lib")
    commandLine(cmd)
}

// Without cargo, a prebuilt libmotoride.so must already be in jniLibs.
val checkNativeLib by tasks.registering {
    dependsOn(cargoNdkBuild)
    doLast {
        val found = fileTree(jniDir) { include("**/libmotoride.so") }.files
        if (found.isEmpty()) {
            throw GradleException(
                "No libmotoride.so found and cargo is not installed here. " +
                "Build the Rust library elsewhere (see README: GitHub Actions) and copy it to " +
                "app/src/main/jniLibs/arm64-v8a/libmotoride.so"
            )
        }
    }
}

tasks.matching { it.name.startsWith("merge") && it.name.endsWith("JniLibFolders") }
    .configureEach { dependsOn(checkNativeLib) }
