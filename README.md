<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/3a215927-ca0b-4246-ae5f-cbab41cbfb2a" /># SOTERIA - Crash Detection Demo

SOTERIA, a Streamlit-based application that uses a YOLOv8 image classification model (in ONNX format) to detect crashes in video frames or images.
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/6296eb2f-c042-476b-95ed-3d45eb1096b6" />

## Project Structure

-   `app/`: Contains the Streamlit web application ([app/app.py](app/app.py)).
-   `crash_classifier/`: Contains scripts for data processing, model training ([crash_classifier/training/train.py](crash_classifier/training/train.py)), and model exporting ([crash_classifier/models/export.py](crash_classifier/models/export.py)).
-   `exports/`: This is where the application expects to find the ONNX model.
-   `runs/`: Default output directory for model training experiments.
-   `requirements.txt`: Python dependencies.

## Setup and First-Time Run

### 1. Prerequisites

-   Python 3.9+
-   Git (for cloning, if haven't already)

### 2. Environment Setup

Use a Python venv (virtual environment) to install SOTERIA dependencies

```bash
# Navigate to project directory (SOTERIA)
cd path/to/SOTERIA

# Create a venv (e.g: .venv/ or venv/)
python -m venv .venv (or python3 on Mac)

# Activate the virtual environment
# Windows:
# .venv\Scripts\activate
# MacOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Placement

The Streamlit application ([app/app.py](app/app.py)) requires a `best.onnx` model file to function.

-   **Create the necessary directory structure**:
    Ensure you have an `exports` directory in the root of your `SOTERIA` project. Inside `exports`, create a subdirectory for your model. For example:
    ```
    SOTERIA/
    ├── exports/
    │   └── final-model/  <-- Or any name for the model version (can be changed)
    │       └── best.onnx
    ...
    ```
-   **Place your model**:
    Copy `best.onnx` file into this subdirectory (e.g., `SOTERIA/exports/final-model/best.onnx`). The application will automatically pick up the most recently modified `best.onnx` file found within any subdirectory of `SOTERIA/exports/`.
    The `best.torchscript` file is not used by the Streamlit application but can be kept alongside for other purposes.

    *If you don't have a `best.onnx` file, you'll need to train a model using [crash_classifier/training/train.py](crash_classifier/training/train.py) and then export it using [crash_classifier/models/export.py](crash_classifier/models/export.py). The export script will typically place the `best.onnx` file in a subdirectory within `crash_classifier/exports/`. You would then need to move it to the `SOTERIA/exports/your_model_directory/` structure mentioned above.*

### 4. Running the Application

Once your environment is set up and the `best.onnx` model is in the correct location, you can run the Streamlit application:

```bash
streamlit run app/app.py
```

This will start the web server, and you can view the application in your browser (usually at `http://localhost:8501`).
