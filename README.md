<h1>🚗 Auto Number Plate Recognition (ANPR) using YOLOv8 + Tesseract OCR</h1>

<p>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python" />
  </a>
  <a href="https://github.com/ultralytics/ultralytics">
    <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-success" alt="YOLOv8" />
  </a>
  <a href="https://github.com/tesseract-ocr/tesseract">
    <img src="https://img.shields.io/badge/OCR-Tesseract-yellow" alt="OCR" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
  </a>
</p>

<blockquote>
  A real-time license plate recognition system using YOLOv8 for detection and Tesseract OCR for text extraction. Designed for vehicle monitoring, surveillance, and traffic analysis.
</blockquote>

<hr />

<h2>📸 Sample Output</h2>

<table border="1" cellpadding="8">
  <thead>
    <tr>
      <th>Detection</th>
      <th>OCR Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/license_plate.png" alt="Detected Plate" width="300"></td>
      <td><strong>MH12AB1234</strong></td>
    </tr>
  </tbody>
</table>

<hr />

<h2>📂 Project Structure</h2>

<pre><code>
anpr-yolo-ocr/
├── data/                  # Datasets
│   ├── train/             # Training images
│   ├── val/               # Validation images
│   └── test/              # Test images
├── model/                 # Pretrained and trained models
├── ocr/                   # OCR text extractor
├── inference/             # Script to test inference
├── notebooks/             # Jupyter notebooks for EDA/training
├── utils/                 # Preprocessing and helpers
├── train.py               # YOLO training script
├── anpr.yaml              # YOLO dataset config
├── requirements.txt       # Dependencies
└── README.md              # You're here!
</code></pre>
