# Hakyking Windows EXE Build

## Short Answer

可以先打包 exe，后面继续修改源码。

但 exe 是一次性构建产物：修改 `hakyking/` 里的源码后，需要重新运行打包脚本，生成新的 `dist/Hakyking/Hakyking.exe`。

## Build Command

```powershell
powershell -ExecutionPolicy Bypass -File .\dev_tools\build_windows_exe.ps1 -Clean
```

临时快速打包，不跑完整 QA：

```powershell
powershell -ExecutionPolicy Bypass -File .\dev_tools\build_windows_exe.ps1 -SkipQa
```

## Output

```text
dist\Hakyking\Hakyking.exe
```

The first supported packaging mode is PyInstaller `onedir`, not `onefile`.
Audio libraries, Qt plugins, Rubber Band, and optional FFmpeg tools are easier
to keep stable in `onedir`.

## Runtime Tools

- Rubber Band is bundled from `tools\rubberband` when that optional directory exists.
- FFmpeg/FFprobe are still expected on system `PATH`, unless a future build adds
  them under `tools\ffmpeg`.

## Recommended Loop

1. Edit source under `hakyking`.
2. Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

3. Rebuild:

```powershell
powershell -ExecutionPolicy Bypass -File .\dev_tools\build_windows_exe.ps1
```

4. Test the new exe from `dist/Hakyking/Hakyking.exe`.
