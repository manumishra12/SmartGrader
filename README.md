# SmartGrader

## 🌟 Overview

SmartGrader is an innovative multimodal AI framework designed to automate the grading of handwritten answer scripts in computer science, particularly focusing on Data Structures and Algorithms (DSA) . Leveraging Vision-Language Models (VLMs) and Large Language Models (LLMs) , this system interprets scanned handwritten responses, evaluates them using domain-specific knowledge, and generates structured feedback aligned with predefined rubrics.

This project proposes an innovative multimodal AI framework designed to handle diverse and complex tasks related to Data Structures and Algorithms (DSA). By leveraging large language models (LLMs), Vision-Language Models (VLMs), and advanced reasoning models, the system aims to create a unified solution capable of processing textual, visual, and structured data. The primary goal is to develop a comprehensive AI-driven DSA tutoring system that can interpret handwritten inputs, analyze problems, and generate solutions with logical reasoning.

## 🎯 Objectives

1. **Automated Handwritten Script Recognition**
   - Extract text from scanned images using Vision-Language Models (e.g., Qwen2.5-VL-3B-Instruct).
   - Preserve structure and context, including flowcharts and equations.

2. **Domain-Specific Evaluation Using LLMs**
   - Fine-tune Google Gemma 3-4B-Instruct model on DSA datasets.
   - Evaluate answers based on rubrics covering clarity, accuracy, depth, structure, and grammar.

3. **Efficient Model Training with LoRA**
   - Use Low-Rank Adaptation (LoRA) to fine-tune large models efficiently on limited hardware.
   - Achieve high performance while minimizing computational overhead.

4. **Integration with RAG for Enhanced Accuracy**
   - Retrieve relevant concepts from a structured knowledge base during evaluation.
   - Improve consistency and reduce bias in automated grading.

5. **Real-World Applicability**
   - Support multilingual input and diverse handwriting styles.
   - Enable deployment in MOOCs, online assessments, and university-level education systems.

---

## 📊 System Architecture

![Proposed System Architecture](https://github.com/manumishra12/SmartGrader/blob/main/assets/architecture.png)

### 🔧 Key Components

#### 1. **Vision-Language Model (Qwen2.5-VL-3B-Instruct)**
- **Input**: Scanned handwritten answer sheets, diagrams, and code snippets.
- **Capabilities**:
  - Dynamic resolution handling for high-quality image processing.
  - Multimodal Rotary Position Embedding (M-RoPE) for spatial understanding.
  - Supports both static and video-based input formats.
- **Output**: Structured text extracted from visual inputs.

#### 2. **Subject Expert LLM (Google Gemma 3-4B-Instruct)**
- **Input**: Transcribed text from VLM.
- **Processing**:
  - Fine-tuned using LoRA on a DSA-specific dataset containing 700+ QA pairs.
  - Rubric-based prompt engineering for clarity, correctness, and depth evaluation.
  - Integration with RAG for contextual knowledge retrieval.
- **Output**: Graded response with detailed feedback stored in JSON format.

#### 3. **Retrieval-Augmented Generation (RAG)**
- Enhances grading accuracy by retrieving best practices and model solutions from a knowledge base.
- Ensures alignment with standard teaching methodologies and reduces overfitting.

#### 4. **Low-Rank Adaptation (LoRA)**
- Efficient parameter tuning technique used to fine-tune both VLM and LLM components.
- Reduces memory usage and training cost significantly compared to full fine-tuning.

#### 5. **Structured Knowledge Base**
- Contains manually curated DSA concepts, time complexities, and algorithm descriptions.
- Used during inference to provide accurate and context-aware evaluations.

---

## 🛠️ Dependencies

### Python Libraries
- `transformers` – For model loading and generation.
- `torch` – For GPU-accelerated inference.
- `opencv-python` – For image preprocessing.
- `json`, `re`, `os`, `random` – For data handling and utilities.
- `IPython.display` – For Jupyter notebook integration.

### Models
- `Qwen2.5-VL-3B-Instruct` – Vision-language model for OCR.
- `Gemma-3-4B-Instruct` – Large language model for evaluation.
- Custom fine-tuned versions hosted on Hugging Face.

### Datasets
- **Handwritten Answer Script Dataset**: 45 samples (40 train + 5 test), varying in quality and complexity.
- **QA Dataset**: 700+ question-answer pairs covering sorting, searching, trees, graphs, recursion, etc.
- **Knowledge Base**: Structured JSON files containing DSA concepts and rubrics.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Hugging Face account for model access

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

##  📚 References
- Wang et al. [2024]. Qwen2.5 VL-3B-Instruct: Enhancing Vision-Language Model's Perception of the World at Any Resolution.
- Yin et al. [2024]. Gemma: Open Models from Google.
- Hu et al. [2021]. LoRA: Low-Rank Adaptation of Large Language Models.
- Baral et al. [2024]. DrawEduMath: A Dataset for Assessing Mathematical Reasoning in Educational Settings.
  
## 📌 Future Work
- Extend support to other technical domains such as Operating Systems, Machine Learning, and Artificial Intelligence.
- Implement real-time interactive tutoring features.
- Develop mobile applications for student engagement.
- Explore federated learning for privacy-preserving updates.

## 📬 Contact
For questions, collaborations, or feature requests:

📧 Email: connectmanumishra12@gmail.com
💻 GitHub: https://github.com/manumishra12/SmartGrader
