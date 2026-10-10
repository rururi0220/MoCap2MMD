# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

is_win = sys.platform.startswith('win')
is_mac = sys.platform == 'darwin'
is_linux = sys.platform.startswith('linux')

block_cipher = None

hidden_imports = [
    'webview',
    'ufbx',
    'numpy',
    'scipy',
    'scipy.spatial.transform',
    'scipy.ndimage',
]

if is_win:
    hidden_imports += [
        'webview.platforms.winforms',
        'webview.platforms.edgechromium',
        'clr',
        'pythonnet',
    ]
elif is_mac:
    hidden_imports += [
        'webview.platforms.cocoa',
        'objc',
        'WebKit',
        'Foundation',
        'AppKit',
    ]
elif is_linux:
    hidden_imports += [
        'webview.platforms.gtk',
        'gi',
    ]

hidden_imports += collect_submodules('webview')

datas = [
    ('frontend', 'frontend'),
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='MoCap2MMD',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True if is_win else False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if is_mac:
    app = BUNDLE(
        exe,
        name='MoCap2MMD.app',
        icon=None,
        bundle_identifier='com.rururi.mocap2mmd',
        info_plist={
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleVersion': '1.0.0',
            'NSHighResolutionCapable': 'True',
            'NSRequiresAquaSystemAppearance': 'False',
        },
    )
