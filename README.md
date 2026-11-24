# MotoRide — Rust + wgpu motorcycle game (Android Studio)

First-person motorcycle view like the reference video: black T-handlebar and red tank in
front of a grey-blue fogged sky and checkered ground. Endless open field with roadside
posts for speed feel (hitting one slows you down). Pure Rust + wgpu, no Java/Kotlin code
(NativeActivity).

## Versions
- Gradle **9.1.0** (gradle/wrapper/gradle-wrapper.properties)
- Android Gradle Plugin **9.0.1** (AGP 9.x is what requires Gradle 9.1+)
- JDK 17+, compileSdk 35, minSdk 26, ABIs: arm64-v8a, x86_64

## One-time setup
1. Install Rust: https://rustup.rs
2. `rustup target add aarch64-linux-android x86_64-linux-android`
3. `cargo install cargo-ndk`
4. Android Studio -> SDK Manager -> SDK Tools -> install **NDK (Side by side)**
5. Make sure `ANDROID_NDK_HOME` points to the installed NDK (cargo-ndk needs it)

## Build & run
Open the `MotoRide` folder in Android Studio, let it sync (it downloads Gradle 9.1.0),
then press Run on a device. Gradle runs `cargo ndk` automatically and packs
`libmotoride.so` into the APK.

If Studio does not create `gradlew` itself, run once: `gradle wrapper --gradle-version 9.1.0`

## Building without a PC (phone IDE such as AndroidIDE)
A phone IDE has no `cargo`, so the Rust library cannot be built there. Use GitHub Actions:
1. Push this folder to a GitHub repository.
2. Open the Actions tab -> "Build APK" -> Run workflow (it also runs on every push).
3. Download the `motoride-debug-apk` artifact and install the APK.

Or: build `libmotoride.so` anywhere with cargo-ndk and copy it to
`app/src/main/jniLibs/arm64-v8a/libmotoride.so`; Gradle then skips the Rust step.
(Remove `app/src/main/jniLibs/` from `.gitignore` if you commit a prebuilt library.)

## Controls (landscape)
- Left half: drag left/right to steer (bike leans)
- Right half, upper: throttle
- Right half, bottom: brake
- Green bar on the tank = speed

## Desktop test
`cd app/rust && cargo run --bin motoride-desktop` (no keyboard controls yet, touch only)
