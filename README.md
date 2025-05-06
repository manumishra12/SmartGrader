# SmartGrader

## 🌟 Overview

This project proposes an innovative multimodal AI framework designed to handle diverse and complex tasks related to Data Structures and Algorithms (DSA). By leveraging large language models (LLMs), Vision-Language Models (VLMs), and advanced reasoning models, the system aims to create a unified solution capable of processing textual, visual, and structured data. The primary goal is to develop a comprehensive AI-driven DSA tutoring system that can interpret handwritten inputs, analyze problems, and generate solutions with logical reasoning.

## 🎯 Objectives

1. **Fine-tune LLMs for DSA Expertise**: Utilize structured JSON files as a knowledge source to train LLMs to become domain experts in DSA.
2. **Integrate VLMs for Visual Understanding**: Employ Qwen-VL models to interpret handwritten inputs such as diagrams, equations, and pseudocode.
3. **Enhance Logical Reasoning**: Incorporate Google's Gemma 3 model to improve instruction-following abilities, context understanding, and deep analytical reasoning.
4. **Create a Seamless Multimodal System**: Combine textual, visual, and structured data processing to solve DSA-related problems effectively.
5. **Revolutionize Educational Tools**: Develop a platform that can assist in coding, problem-solving, and tutoring for DSA concepts.

## 📊 Architecture

The proposed system architecture is depicted in the following diagram:

![Proposed System Architecture](https://github.com/manumishra12/SmartGrader/blob/main/assets/architecture.png)

### Key Components

1. **Vision-Language Model (VLM) for OCR**:
   - **Input**: Scanned answer sheets or handwritten content.
   - **Output**: Extracted text from images using OCR capabilities of the VLM.

2. **Subject Expert LLM for Grading**:
   - **Input**: Extracted text from the VLM.
   - **Processing**:
     - Fine-tuned LLMs trained on DSA-specific knowledge.
     - Prompt engineering to guide the model toward accurate analysis and reasoning.
   - **Output**: Graded output stored in a database.

3. **Retrieval-Augmented Generation (RAG)**:
   - Enhances the LLM's ability to retrieve relevant information from a knowledge base for more accurate and context-aware responses.

4. **Qwen-VL Models**:
   - Handle visual inputs, including diagrams and equations, by converting them into interpretable formats.

5. **Gemma 3 Model**:
   - Strengthens logical reasoning and problem-solving capabilities through advanced instruction-following and expanded context windows.

## 🛠️ Dependencies

To run this project, the following dependencies are required:

1. **Python Libraries**:
   - `transformers` (Hugging Face library for LLMs and VLMs)
   - `torch` or `tensorflow` (for model inference)
   - `opencv-python` (for image processing)
   - `json` (for handling structured data)
   - `pandas` (for data manipulation)

2. **Models**:
   - Pre-trained LLMs (e.g., Qwen series, Gemma 3)
   - Fine-tuned LLMs for DSA expertise
   - VLMs for OCR and visual understanding

3. **Datasets**:
   - Structured JSON files for DSA knowledge
   - Question-Answer datasets for fine-tuning
   - Handwritten input datasets for training VLMs

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- GPU support (recommended for faster inference)
- Hugging Face account (for accessing pre-trained models)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/manumishra12/SmartGrader.git
   cd multimodal-dsa-framework
   ```

2. Install dependencies
   ```bash
     pip install -r requirements.txt
   ```

3. Download pre-trained models:
   Use Hugging Face's transformers library to download models:
   ```bash
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("model_name")
    tokenizer = AutoTokenizer.from_pretrained("model_name")
   ```

4.Prepare datasets:
  Place structured JSON files and question-answer datasets in the data/ directory.
  Ensure scanned answer sheets or handwritten inputs are in the input/ directory.
  
   ```bash
     python ocr.py --input_dir input/ --output_dir extracted_text/
     python llm_inference.py --input_dir extracted_text/ --output_dir graded_output/
     python database.py --input_dir graded_output/
   ```

## 🧩 Contributing
We welcome contributions to enhance the framework! Here are some ways you can contribute:

- Improve Fine-Tuning Processes : Optimize the fine-tuning scripts for better performance.
- Expand Dataset Coverage : Contribute additional DSA problems and solutions for training.
- Enhance Visual Processing : Improve the accuracy of handwritten code recognition.
- Add New Features : Integrate new models or modalities to expand the system's capabilities.
