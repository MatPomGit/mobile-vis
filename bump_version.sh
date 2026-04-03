#!/usr/bin/env bash
# bump_version.sh – increases the app version in android/build.gradle.kts
#
# Usage:
#   ./bump_version.sh          # bumps the patch segment  (1.9.0 → 1.9.1)
#   ./bump_version.sh patch    # same as above
#   ./bump_version.sh minor    # bumps the minor segment  (1.9 → 1.10, resets patch)
#   ./bump_version.sh major    # bumps the major segment  (1.9 → 2.0, resets minor+patch)
#
# The script updates two lines in android/build.gradle.kts:
#   val app_version_name by extra("X.Y[.Z]")
#   val app_version_code by extra(N)

set -euo pipefail

GRADLE_FILE="$(dirname "$0")/android/build.gradle.kts"
BUMP_TYPE="${1:-patch}"

if [[ ! -f "$GRADLE_FILE" ]]; then
    echo "Error: $GRADLE_FILE not found." >&2
    exit 1
fi

# ---------- read current values -----------------------------------------------
CURRENT_NAME=$(grep -E 'val app_version_name by extra' "$GRADLE_FILE" \
    | sed -E 's/.*"([^"]+)".*/\1/')
CURRENT_CODE=$(grep -E 'val app_version_code by extra' "$GRADLE_FILE" \
    | sed -E 's/[^0-9]*([0-9]+).*/\1/')

if [[ -z "$CURRENT_NAME" ]]; then
    echo "Error: could not read app_version_name from $GRADLE_FILE." >&2
    exit 1
fi
if [[ -z "$CURRENT_CODE" ]]; then
    echo "Error: could not read app_version_code from $GRADLE_FILE." >&2
    exit 1
fi

# ---------- parse version segments --------------------------------------------
IFS='.' read -ra PARTS <<< "$CURRENT_NAME"
MAJOR="${PARTS[0]:-1}"
MINOR="${PARTS[1]:-0}"
PATCH="${PARTS[2]:-}"   # may be empty for two-segment versions like "1.9"

# ---------- bump the requested segment ----------------------------------------
case "$BUMP_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=""
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=""
        ;;
    patch)
        if [[ -z "$PATCH" ]]; then
            # Promote two-segment version to three-segment and bump patch.
            PATCH=1
        else
            PATCH=$((PATCH + 1))
        fi
        ;;
    *)
        echo "Error: unknown bump type '$BUMP_TYPE'. Use major, minor, or patch." >&2
        exit 1
        ;;
esac

# ---------- build new version string ------------------------------------------
if [[ -z "$PATCH" ]]; then
    NEW_NAME="${MAJOR}.${MINOR}"
else
    NEW_NAME="${MAJOR}.${MINOR}.${PATCH}"
fi

NEW_CODE=$((CURRENT_CODE + 1))

# ---------- update the file ---------------------------------------------------
# Use a temp file to avoid in-place issues on macOS and Linux alike.
TMP=$(mktemp)
sed \
    -e "s|val app_version_name by extra(\"[^\"]*\")|val app_version_name by extra(\"${NEW_NAME}\")|" \
    -e "s|val app_version_code by extra(${CURRENT_CODE})|val app_version_code by extra(${NEW_CODE})|" \
    "$GRADLE_FILE" > "$TMP"
mv "$TMP" "$GRADLE_FILE"

echo "Version bumped: ${CURRENT_NAME} → ${NEW_NAME}  (code: ${CURRENT_CODE} → ${NEW_CODE})"
