# Android Signing Configuration

This document describes how to configure Android release signing locally and in CI.

---

## Supported properties

The following values are read first from **Gradle properties** and then from **environment
variables** (whichever is non-blank wins):

| Gradle property            | Environment variable           | Description                              |
|----------------------------|-------------------------------|------------------------------------------|
| `ANDROID_KEYSTORE_PATH`    | `ANDROID_KEYSTORE_PATH`       | Absolute path to the `.jks` / `.keystore` file |
| `ANDROID_KEYSTORE_PASSWORD`| `ANDROID_KEYSTORE_PASSWORD`   | Password for the keystore                |
| `ANDROID_KEY_ALIAS`        | `ANDROID_KEY_ALIAS`           | Alias of the signing key inside the keystore |
| `ANDROID_KEY_PASSWORD`     | `ANDROID_KEY_PASSWORD`        | Password for the key                     |

If `ANDROID_KEYSTORE_PATH` is absent or blank, the release signing config is left
unconfigured.  Debug builds always use the standard Android debug keystore and are unaffected.

---

## Local development

Add the properties to your **personal** Gradle properties file so that they are never committed
to the repository:

```properties
# ~/.gradle/gradle.properties  (do NOT add to the project gradle.properties)
ANDROID_KEYSTORE_PATH=/home/yourname/.android/release.jks
ANDROID_KEYSTORE_PASSWORD=your_store_password
ANDROID_KEY_ALIAS=your_key_alias
ANDROID_KEY_PASSWORD=your_key_password
```

Then build a release APK normally:

```bash
cd android
./gradlew assembleRelease
```

---

## GitHub Actions (CI)

Store the secrets in **GitHub → repository Settings → Secrets and variables → Actions**:

| Secret name                 | Value                                            |
|-----------------------------|--------------------------------------------------|
| `KEYSTORE_BASE64`           | Base64-encoded keystore file (generate with: `base64 --wrap=0 release.jks`) |
| `ANDROID_KEYSTORE_PASSWORD` | Keystore password                               |
| `ANDROID_KEY_ALIAS`         | Key alias                                       |
| `ANDROID_KEY_PASSWORD`      | Key password                                    |

Then add a release build job (see the commented template in
`.github/workflows/android.yml`):

```yaml
- name: Decode keystore
  run: echo "${{ secrets.KEYSTORE_BASE64 }}" | base64 --decode > android/app/release.jks

- name: Build Release APK
  env:
    ANDROID_KEYSTORE_PATH: ${{ github.workspace }}/android/app/release.jks
    ANDROID_KEYSTORE_PASSWORD: ${{ secrets.ANDROID_KEYSTORE_PASSWORD }}
    ANDROID_KEY_ALIAS: ${{ secrets.ANDROID_KEY_ALIAS }}
    ANDROID_KEY_PASSWORD: ${{ secrets.ANDROID_KEY_PASSWORD }}
  run: ./gradlew assembleRelease --stacktrace
  working-directory: ./android
```

---

## Security notes

* **Never** commit plaintext passwords or `.jks` files to the repository.
* `android/app/release.jks` (the decoded keystore written during CI) is listed in
  `.gitignore` so it cannot be accidentally committed.
* Rotate secrets immediately if they are ever exposed in logs or committed by mistake.
