# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path(SPECPATH)

datas = []
bundled_rubberband = ROOT / "tools" / "rubberband"
if bundled_rubberband.exists():
    datas.append((str(bundled_rubberband), "tools/rubberband"))

hiddenimports = [
    "audioflux",
    "librosa",
    "parselmouth",
    "pyrubberband",
    "pyworld",
    "sounddevice",
    "soundfile",
]

a = Analysis(
    [str(ROOT / "hakyking" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "dev_tools",
        "matplotlib",
        "onnxruntime",
        "pytest",
        "qa_artifacts",
        "rmvpe_onnx",
        "tensorflow",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Hakyking",
    icon=str(ROOT / "assets" / "hakyking.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Hakyking",
)
