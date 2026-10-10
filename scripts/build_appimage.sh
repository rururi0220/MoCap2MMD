#!/usr/bin/env bash
set -e

# ==============================================================================
# MoCap2MMD Linux AppImage Builder
# ==============================================================================

echo ">>> [1/5] Checking environment and dependencies..."
python3 -m pip install -r requirements.txt pyinstaller "PyGObject<3.52.0" || true

echo ">>> [2/5] Building standalone ELF binary with PyInstaller..."
python3 -m PyInstaller MoCap2MMD.spec --noconfirm --clean

if [ ! -f "dist/MoCap2MMD" ]; then
    echo "ERROR: dist/MoCap2MMD not found! PyInstaller build failed."
    exit 1
fi

echo ">>> [3/5] Constructing AppDir structure..."
rm -rf AppDir
mkdir -p AppDir/usr/bin
mkdir -p AppDir/usr/share/icons/hicolor/256x256/apps
mkdir -p AppDir/usr/share/applications

cp dist/MoCap2MMD AppDir/usr/bin/MoCap2MMD
chmod +x AppDir/usr/bin/MoCap2MMD

# AppRun launcher
cat << 'EOF' > AppDir/AppRun
#!/bin/sh
SELF=$(readlink -f "$0")
HERE=${SELF%/*}
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
exec "${HERE}/usr/bin/MoCap2MMD" "$@"
EOF
chmod +x AppDir/AppRun

# Desktop specification
cat << 'EOF' > AppDir/MoCap2MMD.desktop
[Desktop Entry]
Name=MoCap2MMD
Exec=MoCap2MMD
Icon=mocap2mmd
Type=Application
Categories=Graphics;3DGraphics;AudioVideo;
Comment=FBX / BVH to MMD VMD Motion Retargeting Tool
Terminal=false
StartupNotify=true
EOF
cp AppDir/MoCap2MMD.desktop AppDir/usr/share/applications/

# Icon integration
if [ -f "frontend/assets/icon.png" ]; then
    cp frontend/assets/icon.png AppDir/mocap2mmd.png
    cp frontend/assets/icon.png AppDir/usr/share/icons/hicolor/256x256/apps/mocap2mmd.png
    ln -sf mocap2mmd.png AppDir/.DirIcon
fi

echo ">>> [4/5] Fetching appimagetool..."
if [ ! -f "appimagetool-x86_64.AppImage" ]; then
    wget -q https://github.com/AppImage/AppImageKit/releases/download/13/appimagetool-x86_64.AppImage
    chmod +x appimagetool-x86_64.AppImage
fi

echo ">>> [5/5] Packaging AppImage..."
export ARCH=x86_64
./appimagetool-x86_64.AppImage --appimage-extract-and-run AppDir dist/MoCap2MMD-x86_64.AppImage

echo "=================================================================="
echo " SUCCESS: AppImage created at dist/MoCap2MMD-x86_64.AppImage"
echo "=================================================================="
