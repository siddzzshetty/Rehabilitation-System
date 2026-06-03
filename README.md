# Rehabilitation System

AI-assisted physiotherapy system for:
- Prescription OCR (image/PDF) to identify prescribed exercises
- Real-time pose tracking via webcam
- Sequence-model based form validation (LSTM-CNN hybrid, TorchScript)
- LLM-generated corrective feedback
- Session performance visualization

---

## 1) What this project does

This project guides a patient through rehabilitation exercises by combining computer vision, time-series modeling, and LLM feedback.

High-level flow:
1. User uploads prescription (image/PDF).
2. OCR extracts text and detects exercise names.
3. User selects exercise and starts webcam session.
4. Pose landmarks are converted to biomechanical angles.
5. Model predicts short-horizon future movement.
6. Predicted vs actual motion error is used to detect poor form.
7. If poor form persists, LLM generates concise correction feedback.
8. Session metrics are visualized in graphs.

---

## 2) Tech stack

### Core
- Python 3.10+
- Streamlit (UI)
- PyTorch (model inference/training)
- Mediapipe (pose landmarks)
- OpenCV (camera + image preprocessing)
- NumPy, Pandas, scikit-learn

### OCR
- Google Cloud Vision API
- PyMuPDF (`fitz`) for PDF text extraction
- `pdf2image` fallback for scanned PDFs
- Pillow
- fuzzywuzzy (exercise keyword matching)

### LLM
- LangChain + Groq (`langchain_groq`)

### Visualization
- Matplotlib
- TensorBoard logs (training runs in `runs/`)

---

## 3) Repository structure (important files)

### Runtime app
- `login.py`  
	Entry page. Handles authentication UI, OCR, prescription parsing, exercise detection, and navigation to exercise page.

- `pages/final1.py`  
	Main exercise session page (camera start/stop, instructions, calls real-time pipeline).

- `st_helper.py`  
	Core runtime logic:
	- pose angle computation
	- real-time sequence buffering
	- model loading + prediction
	- error thresholding
	- feedback storage/rendering helpers
	- sidebar navigation

- `config.py`  
	LLM model setup, prompts, exercise map, runtime flags.

- `pages/chatbot_page.py`  
	General safety-restricted chatbot page.

- `pages/chatbot_display.py`  
	Displays historical corrective feedback (`llm_feedback.pkl`).

- `pages/graphs.py`  
	Session-wise accuracy visualization from `session_accuracy.csv`.

### Model + data artifacts
- `model_path/*.pt`  
	Trained/scripted models used during runtime.

- `data/*.csv`  
	Angle datasets and labels used in training/experiments.

- `model.ipynb`, `backend.ipynb`, `model_testing.ipynb`  
	Training, experiments, and evaluation notebooks.

---

## 4) Setup

## 4.1 Create environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

## 4.2 Install dependencies

`requirements.txt` currently contains only part of runtime dependencies (mainly torch/tensorboard). Install required packages:

```bash
pip install -r requirements.txt
pip install streamlit opencv-python numpy pandas scikit-learn mediapipe pillow pymupdf pdf2image fuzzywuzzy python-Levenshtein langchain langchain-core langchain-groq google-cloud-vision matplotlib nest_asyncio
```

> Note: `pdf2image` may require Poppler binaries installed on your OS and available in PATH.

## 4.3 Configure credentials and keys

### Google Vision OCR
- Place your GCP service account JSON in project root.
- Set env var:

```bash
# Windows PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS="carbon-pride-453005-g3-2b983e32ddd4.json"
```

### Groq API

```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key"
```

> Important: `config.py` expects `GROQ_API_KEY` to be available.

---

## 5) Run the application

From project root:

```bash
streamlit run login.py
```

Then:
1. Upload prescription image/PDF.
2. Select detected exercise.
3. Start camera.
4. Review live feedback and session pages from sidebar.

---

## 6) Detailed runtime workflow

## 6.1 OCR + exercise extraction

- `login.py` preprocesses image:
	- resize very large image
	- grayscale conversion
	- Gaussian blur
	- Otsu thresholding
- OCR using Google Vision API.
- For PDFs:
	1. Try direct text extraction using PyMuPDF.
	2. If no embedded text, render PDF pages to images and OCR each page.
- Parsed lines are fuzzy-matched against exercise keywords:
	- `squat`, `pushup`, `jumping jack`, `sit up`, `pull-up`

## 6.2 Real-time computer vision pipeline

Inside `st_helper.py::run_camera_feed`:
1. Open webcam via OpenCV.
2. Run Mediapipe Pose per frame.
3. Compute 7 biomechanical angles from landmark triplets.
4. Normalize angles with MinMax scaling.
5. Push angles into rolling input window.

## 6.3 Sequence prediction and error logic

Runtime constants:
- `INPUT_WINDOW = 20`
- `OUTPUT_WINDOW = 5`
- `PRED_FREQ = 5`
- `THRESHOLD = 300`
- `BUFFER_LEN = 5`
- `LLM_COOLDOWN = 10`

Process:
1. When 20-frame window is ready, model predicts next 5 frames.
2. System collects actual 5 future frames.
3. Computes MSE between predicted vs actual focus angles.
4. Maintains short loss buffer.
5. If all recent losses exceed threshold, marks likely poor form.

## 6.4 LLM feedback generation

- Uses exercise-specific prompt templates (`config.py`).
- Sends recent predicted/actual angle runs as context.
- LLM returns concise:
	- `Incorrect: ...`
	- `Suggestion: ...`
- Feedback saved in `llm_feedback.pkl` and shown in UI.

---

## 7) Feature engineering (what and why)

Instead of using raw pixels, this project uses **joint angles** as features.

### Why angle features?
- Lower dimensionality than image frames.
- More robust to lighting/background noise.
- Biomechanically meaningful and interpretable.
- Better suited for rehabilitation movement quality analysis.

### Angle computation

For three points `a, b, c`, angle at `b` is derived from vector cosine relation:

\[
\cos\theta = \frac{(b-a)\cdot(c-b)}{\|b-a\|\|c-b\|}
\]

Implemented in `calculate_angle(...)` and transformed with `180 - theta` for chosen convention.

---

## 8) LSTM-CNN model methodology

The notebook (`model.ipynb`) defines a hybrid architecture:

1. LSTM layer to capture temporal dependencies.
2. Dense bottleneck/projection layers.
3. Second LSTM refinement stage.
4. 1D convolution across temporal axis.
5. Final dense projection to output window `(5 x 7)`.

### Why hybrid LSTM + Conv1D?
- LSTM captures long-term movement dynamics.
- Conv1D captures local temporal patterns and smooth transitions.
- Useful for short-horizon trajectory forecasting in exercise motion.

### Training setup (from notebook)
- Input window: 20
- Output window: 5
- Features: 7
- Train/test split: 80/20
- Optimizer: AdamW
- Loss: HuberLoss
- Epochs: 400 (experiment setting)
- Optional TensorBoard logging in `runs/`

Models are exported as TorchScript `.pt` and loaded at runtime with `torch.jit.load(...)`.

---

## 9) Current exercise support in runtime

In live inference mapping, current wired models are:
- squat
- pushup

Other exercise model files exist in `model_path/`, but page logic may need extension to fully route and evaluate them in the same pipeline.

---

## 10) Known caveats

- `requirements.txt` is incomplete for full runtime stack.
- Some imports in runtime pages are unused/redundant (experimental leftovers).
- `session_accuracy.csv` graphing exists, but session logging calls may need alignment depending on current branch state.
- Service account JSON should not be committed in public repositories.

---

## 11) Suggested next improvements

1. Fully pin all runtime dependencies in `requirements.txt`.
2. Move all secrets/keys to `.env` and read via `os.getenv`.
3. Add explicit support paths for pull-up/sit-up/jumping-jack in runtime model map.
4. Add unit tests for angle extraction and sequence windowing.
5. Add a clean architecture diagram in docs.

---

## 12) Quick demo checklist

1. Set `GOOGLE_APPLICATION_CREDENTIALS` and `GROQ_API_KEY`.
2. `streamlit run login.py`.
3. Upload a prescription.
4. Select detected exercise.
5. Start camera and observe live corrective feedback.
6. Open sidebar pages for feedback history and performance graph.