#!/usr/bin/env bash
set -e

# ==============================================================================
# MoCap2MMD macOS Application & DMG Builder
# ==============================================================================

echo ">>> [1/4] Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt pyinstaller pyobjc-framework-WebKit

echo ">>> [2/4] Building macOS App bundle with PyInstaller..."
python3 -m PyInstaller MoCap2MMD.spec --noconfirm --clean

if [ ! -d "dist/MoCap2MMD.app" ]; then
    echo "ERROR: dist/MoCap2MMD.app not found! PyInstaller build failed."
    exit 1
fi

echo ">>> [3/4] Creating ZIP archive..."
cd dist
zip -r -q "MoCap2MMD-macOS.zip" "MoCap2MMD.app"
cd ..

echo ">>> [4/4] Creating DMG disk image..."
if command -v hdiutil &> /dev/null; then
    hdiutil create -volname "MoCap2MMD" -srcfolder "dist/MoCap2MMD.app" -ov -format UDZO "dist/MoCap2MMD-macOS.dmg"
    echo "=================================================================="
    echo " SUCCESS: DMG created at dist/MoCap2MMD-macOS.dmg"
    echo "=================================================================="
else
    echo "Notice: hdiutil not available; zip archive available at dist/MoCap2MMD-macOS.zip"
fi
