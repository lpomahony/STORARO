# STORARO (Scene Topography and Object Recognition for Aesthetic Reconstruction and Output)

This project is a portfolio project demonstrating the use of state-of-the-art computer vision techniques to analyze film stills. It combines object detection (using YOLOv8) and monocular depth estimation (using Marigold) to extract information about scene composition, both in 2D and 3D.

## Current Features (MVP)

*   **Object Detection:** Uses a pre-trained YOLOv8 model to detect objects within a film still and draw bounding boxes around them.
*   **Depth Estimation:** Employs the Marigold depth estimation model (from Hugging Face Diffusers) to generate a depth map from a single film still.
*   **2D Composition Analysis:**
    *   **Rule of Thirds:**  Overlays a rule-of-thirds grid on the image.
    *   **Object Size and Placement:**  Analyzes the relative size (Large, Medium, Small) and screen position (Top-Left, Center-Right, etc.) of detected objects.
*   **Depth Layer Analysis:**  Classifies detected objects into depth layers (Foreground, Midground, Background) based on the estimated depth map.
*   **Interactive Visualization:** Uses Streamlit to create a web application where users can upload a film still and see the analysis results (object detection, depth map, textual analysis).
*   **Colorbar:**  Displays a colorbar alongside the depth map to indicate relative depth values.

## Project Structure

*   `app.py`: The main Streamlit application file.  Contains all the logic for image loading, object detection, depth estimation, analysis, and visualization.
*   `test_marigold.py`: A script used for initial testing and debugging of the Marigold depth estimation model (now that the MVP is working, this could be considered a supplementary file).
*   `images/`: A directory to store test film still images.
*   `.gitignore`:  Specifies files and directories that should be ignored by Git.
*   `.venv/`:  (Ignored by Git) The virtual environment directory.
*   `yolov8n.pt`: The pre-trained YOLOv8 model.
*   `.python-version`: The file that tells pyenv to use the local python version.

## Setup and Running the App

1.  **Prerequisites**

    *   Python 3.10: This project was developed and tested with Python 3.10. Using `pyenv` is highly recommended for managing Python versions.

    *   `pyenv` (Recommended): Install pyenv to manage Python installations cleanly.

        ```bash
        curl https://pyenv.run | bash
        ```

    *   Add the following to your `.bashrc` or `.zshrc`, then open a new terminal

        ```bash
        export PYENV_ROOT="$HOME/.pyenv"
        command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
        ```

    *   Install python 3.10.13

        ```bash
        pyenv install 3.10.13
        pyenv local 3.10.13
        ```

2.  **Clone the Repository:**

    ```bash
    git clone <your-github-repository-url>  # You'll get this URL after creating the repo
    cd film_still_analysis
    ```

3.  **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

4.  **Install Dependencies:**

    ```bash
    pip install opencv-python-headless torch torchvision torchaudio ultralytics transformers matplotlib streamlit diffusers accelerate numpy==1.26.4
    ```

5.  **Run the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

    This will start the Streamlit development server, and you can access the app in your web browser (usually at `http://localhost:8501`).

## Future Enhancements (Beyond MVP)

*   **3D Reconstruction:** Generate point clouds or meshes from the depth maps for 3D visualization of scene composition.
*   **Shot Type Classification:** Automatically classify shots (close-up, medium shot, long shot).
*   **Camera Movement Analysis:**  Extend the analysis to short video clips to detect camera movement (pan, tilt, zoom).
*   **Lighting Analysis:** Infer lighting direction and intensity from color and depth information.
*   **Expanded Compositional Analysis:** Detect more advanced compositional elements (leading lines, symmetry, etc.).
*   **Database Integration:** Store analysis results in a database for larger-scale film analysis.
*   **User Authentication:** Add user accounts and authentication for a multi-user application.
*   **Deployment:** Deploy the app to a cloud platform (e.g., Streamlit Cloud, Heroku, AWS, GCP) for wider accessibility.

## Troubleshooting

*   **`ValueError: Unrecognized model in ...`:**  This error *should* no longer occur with the current code, as it was caused by using the wrong API (`transformers` instead of `diffusers`) for the Marigold model.  If you encounter this error, double-check that you are using the `diffusers` API correctly (as shown in `app.py`) and that you have a clean Python 3.10 environment with the correct dependencies installed.
