#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/package-macos.sh [--skip-build]

Builds the macOS desktop app and DMG:
  1. processes macos/icon.png into transparent rounded app icons;
  2. builds target/release/transformer unless --skip-build is used;
  3. creates dist/Transformer.app;
  4. creates dist/Transformer.app.zip;
  5. creates dist/Transformer.dmg and verifies it with hdiutil.

Environment overrides:
  APP_NAME=Transformer
  BINARY_NAME=transformer
  BUNDLE_ID=com.transformer.surrogate
  ICON_PNG=macos/icon.png
  ICON_CORNER_RADIUS_RATIO=0.223
  DIST_DIR=dist
USAGE
}

for arg in "$@"; do
  case "$arg" in
    --help|-h)
      usage
      exit 0
      ;;
    --skip-build)
      SKIP_BUILD=1
      ;;
    *)
      echo "error: unknown argument: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

APP_NAME="${APP_NAME:-Transformer}"
BINARY_NAME="${BINARY_NAME:-transformer}"
BUNDLE_ID="${BUNDLE_ID:-com.transformer.surrogate}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY_PATH="${BINARY_PATH:-$ROOT_DIR/target/release/$BINARY_NAME}"
PACKAGE_VERSION="$(awk -F '"' '/^version = / { print $2; exit }' "$ROOT_DIR/Cargo.toml")"
MACOS_DIR="$ROOT_DIR/macos"
ICON_PNG="${ICON_PNG:-$MACOS_DIR/icon.png}"
ICON_ROUNDED_PNG="${ICON_ROUNDED_PNG:-$MACOS_DIR/icon-rounded.png}"
ICON_RUNTIME_PNG="${ICON_RUNTIME_PNG:-$MACOS_DIR/icon-runtime.png}"
ICON_ICNS="${ICON_ICNS:-$MACOS_DIR/icon.icns}"
ICON_CORNER_RADIUS_RATIO="${ICON_CORNER_RADIUS_RATIO:-0.223}"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist}"
APP_DIR="$DIST_DIR/$APP_NAME.app"
ZIP_PATH="$DIST_DIR/$APP_NAME.app.zip"
DMG_PATH="$DIST_DIR/$APP_NAME.dmg"

if [[ ! -f "$ICON_PNG" ]]; then
  echo "error: source icon not found: $ICON_PNG" >&2
  exit 1
fi

if [[ -z "$PACKAGE_VERSION" ]]; then
  echo "error: package version not found in Cargo.toml" >&2
  exit 1
fi

mkdir -p "$DIST_DIR" "$MACOS_DIR"
rm -rf "$APP_DIR" "$ZIP_PATH" "$DMG_PATH"

if [[ -f "$ICON_PNG" ]]; then
  python3 - "$ICON_PNG" "$ICON_ROUNDED_PNG" "$ICON_RUNTIME_PNG" "$ICON_ICNS" "$ICON_CORNER_RADIUS_RATIO" <<'PY'
import sys
from collections import deque
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter
except ModuleNotFoundError:
    print("error: python3 Pillow package is required to convert icon.png to icon.icns", file=sys.stderr)
    sys.exit(1)

src = Path(sys.argv[1])
rounded_dst = Path(sys.argv[2])
runtime_dst = Path(sys.argv[3])
icns_dst = Path(sys.argv[4])
corner_radius_ratio = float(sys.argv[5])
if not 0 < corner_radius_ratio < 0.5:
    print("error: ICON_CORNER_RADIUS_RATIO must be between 0 and 0.5", file=sys.stderr)
    sys.exit(1)

def rounded_mask(size, ratio):
    scale = 4
    mask_size = size * scale
    mask = Image.new("L", (mask_size, mask_size), 0)
    draw = ImageDraw.Draw(mask)
    radius = int(round(size * ratio * scale))
    draw.rounded_rectangle((0, 0, mask_size - 1, mask_size - 1), radius=radius, fill=255)
    return mask.resize((size, size), Image.Resampling.LANCZOS)

def background_cutout_alpha(image):
    rgb = image.convert("RGB")
    pix = rgb.load()
    w, h = image.size
    seen = bytearray(w * h)
    bg = bytearray(w * h)
    q = deque()

    def idx(x, y):
        return y * w + x

    def looks_like_light_background(x, y):
        r, g, b = pix[x, y]
        brightness = (r + g + b) / 3.0
        saturation = max(r, g, b) - min(r, g, b)
        return brightness >= 145 and saturation <= 72

    for x in range(w):
        q.append((x, 0))
        q.append((x, h - 1))
    for y in range(1, h - 1):
        q.append((0, y))
        q.append((w - 1, y))

    while q:
        x, y = q.popleft()
        i = idx(x, y)
        if seen[i]:
            continue
        seen[i] = 1
        if not looks_like_light_background(x, y):
            continue
        bg[i] = 255
        if x > 0:
            q.append((x - 1, y))
        if x + 1 < w:
            q.append((x + 1, y))
        if y > 0:
            q.append((x, y - 1))
        if y + 1 < h:
            q.append((x, y + 1))

    bg_mask = Image.frombytes("L", (w, h), bytes(bg)).filter(
        ImageFilter.GaussianBlur(max(3, w // 180))
    )
    return ImageChops.subtract(Image.new("L", (w, h), 255), bg_mask)

image = Image.open(src).convert("RGBA")
side = max(image.size)
source = Image.new("RGBA", (side, side), (0, 0, 0, 0))
source.alpha_composite(image, ((side - image.width) // 2, (side - image.height) // 2))

foreground_alpha = ImageChops.multiply(source.getchannel("A"), background_cutout_alpha(source))
edge_mask = rounded_mask(side, corner_radius_ratio)
canvas = source.copy()
canvas.putalpha(ImageChops.multiply(foreground_alpha, edge_mask))

rounded_dst.parent.mkdir(parents=True, exist_ok=True)
canvas.save(rounded_dst)
runtime_icon = canvas.resize((1024, 1024), Image.Resampling.LANCZOS)
runtime_icon.save(runtime_dst)
runtime_icon.save(
    icns_dst,
    format="ICNS",
    sizes=[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
)
PY
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  cargo build --release
fi

if [[ ! -x "$BINARY_PATH" ]]; then
  echo "error: built binary not found or not executable: $BINARY_PATH" >&2
  echo "Build it first, for example: cargo build --release" >&2
  exit 1
fi

mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "$BINARY_PATH" "$APP_DIR/Contents/MacOS/$APP_NAME"
chmod 755 "$APP_DIR/Contents/MacOS/$APP_NAME"
cp "$ICON_ICNS" "$APP_DIR/Contents/Resources/icon.icns"

cat > "$APP_DIR/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDisplayName</key>
	<string>$APP_NAME</string>
	<key>CFBundleExecutable</key>
	<string>$APP_NAME</string>
	<key>CFBundleIconFile</key>
	<string>icon.icns</string>
	<key>CFBundleIdentifier</key>
	<string>$BUNDLE_ID</string>
	<key>CFBundleName</key>
	<string>$APP_NAME</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>$PACKAGE_VERSION</string>
	<key>CFBundleVersion</key>
	<string>$PACKAGE_VERSION</string>
	<key>LSMinimumSystemVersion</key>
	<string>11.0</string>
	<key>NSHighResolutionCapable</key>
	<true/>
</dict>
</plist>
PLIST

plutil -lint "$APP_DIR/Contents/Info.plist" >/dev/null
xattr -cr "$APP_DIR" 2>/dev/null || true

(
  cd "$DIST_DIR"
  COPYFILE_DISABLE=1 zip -qry -X "$APP_NAME.app.zip" "$APP_NAME.app"
)

DMG_STAGE="$(mktemp -d /tmp/transformer-dmg.XXXXXX)"
cleanup() {
  rm -rf "$DMG_STAGE"
}
trap cleanup EXIT

cp -R "$APP_DIR" "$DMG_STAGE/"
ln -s /Applications "$DMG_STAGE/Applications"
COPYFILE_DISABLE=1 hdiutil create -volname "$APP_NAME" -srcfolder "$DMG_STAGE" -fs HFS+ -format UDZO -ov "$DMG_PATH" >/dev/null
hdiutil verify "$DMG_PATH" >/dev/null

echo "App: $APP_DIR"
echo "Zip: $ZIP_PATH"
echo "DMG: $DMG_PATH"
