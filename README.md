# wmbench: An Open-Source Benchmarking Toolkit for Efficient LLM Watermark Generation and Detection at Scale.

## 🚧 Work in Progress

This project is under active development. We are continuously integrating new watermarking schemes and evaluation metrics.

## 💡 Key Features
- High Scalability: Designed with a modular architecture that allows for easy integration of custom watermarking schemes and evaluation metrics.

- (TBD)Extensive Baseline Support: Planned support for a wide range of state-of-the-art (SOTA) watermarking algorithms (e.g., KGW, Unforgeable Watermarks).

- Optimized Performance: High-efficiency parallel generation and detection, specifically tailored for large-scale benchmarking tasks.

## 🛠️ Currently Supported Algorithms

* **KGW**: [Kirchenbauer et al., 2023] A Watermark for Large Language Models.

---

## 🚀 How to Use

This project uses **[uv](https://github.com/astral-sh/uv)** for extremely fast Python package and environment management.

### 1. Prerequisites

Ensure you have `uv` installed. 

### 2. Environment Setup

Clone the repository and sync the dependencies:

```bash
# Create virtual environment and install dependencies automatically
uv sync
```

### 3. Running Benchmarks

The primary entry point for large-scale evaluation is `batch_benchmark.py`. This script allows you to run watermarking generation and detection across various configurations.

To execute a batch benchmark task, use:

```bash
# Using 'uv run' to ensure the environment is correctly loaded
uv run batch_benchmark.py 
```

*(Note: You can customize the parameters in `batch_benchmark.py` or via command-line arguments to suit your experimental setup.)*

---

## 📈 Roadmap

* [ ] Add support for **more** watermarking schemes.
* [ ] Integrate **more**  detection metrics (e.g., ROC/AUC analysis).
* [ ] Comprehensive **documentation** for custom algorithm integration.

## 🙏 Acknowledgements

This project is inspired by [MarkLLM](https://github.com/THU-BPM/MarkLLM). 

If you find this project helpful, a star would be greatly appreciated!
