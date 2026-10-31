# Third-party notices

MangaTL is built on the work of the projects below. Each remains the property
of its authors and is used under its own license. Follow the links for the
full license text.

## AI models

| Component | Used for | Source |
|---|---|---|
| comic-text-and-bubble-detector (ogkalu) | Finding text blocks and balloons | https://huggingface.co/ogkalu/comic-text-and-bubble-detector |
| comic-text-detector (dmMaze) | Glyph mask and text lines | https://github.com/dmMaze/comic-text-detector |
| baberu-ocr (genshiai-daichi) | Reading comic lettering | https://huggingface.co/genshiai-daichi/baberu-ocr |
| LaMa (Samsung AI / advimman) | Erasing text over artwork | https://github.com/advimman/lama |
| lama-manga-onnx (mayocream) | LaMa for manga, ONNX export | https://huggingface.co/mayocream/lama-manga-onnx |
| LaMa-ONNX (Carve) | LaMa, ONNX export | https://huggingface.co/Carve/LaMa-ONNX |
| Gemma (`gemma4:e2b`, Google) | Offline translation; not bundled, downloaded by the user through Ollama | https://ollama.com/library/gemma4 |

## Libraries and runtimes

| Component | License | Source |
|---|---|---|
| Qt 6 | LGPL v3 | https://www.qt.io |
| PyQt6 (Riverbank Computing) | GPL v3 / commercial | https://www.riverbankcomputing.com/software/pyqt/ |
| ONNX Runtime (Microsoft) | MIT | https://github.com/microsoft/onnxruntime |
| NVIDIA CUDA and cuDNN runtime libraries | NVIDIA software license | https://developer.nvidia.com/cuda-toolkit |
| OpenCV | Apache 2.0 | https://opencv.org |
| Skia / skia-python | BSD 3-Clause | https://github.com/kyamagu/skia-python |
| NumPy | BSD 3-Clause | https://numpy.org |
| httpx | BSD 3-Clause | https://www.python-httpx.org |
| pydantic | MIT | https://docs.pydantic.dev |
| PyYAML | MIT | https://pyyaml.org |
| psutil | BSD 3-Clause | https://github.com/giampaolo/psutil |
| nvidia-ml-py | BSD | https://pypi.org/project/nvidia-ml-py/ |
| Python | PSF License | https://www.python.org |
| PyInstaller bootloader | GPL v2 with bootloader exception | https://pyinstaller.org |
| Inno Setup | Inno Setup License | https://jrsoftware.org/isinfo.php |

## Fonts

| Font | Author | License |
|---|---|---|
| Anime Ace 2 | Blambot (Nate Piekos) | Blambot free font license, see https://blambot.com |
| Comic Neue | Craig Rozynski and Hrant Papazian | SIL Open Font License 1.1 |

## Optional external tools

| Tool | Used for |
|---|---|
| Ollama | Running the offline translation model |
| 7-Zip / bsdtar / unrar | Opening `.rar` and `.7z` chapter archives, when installed |
