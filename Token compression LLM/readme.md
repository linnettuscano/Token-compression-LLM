# Token Compression LLM

## Project Overview

The goal of this project is to reproduce the system described in ["TCRA-LLM: Token Compression Retrieval Augmented Large Language Model for Inference Cost Reduction."](https://arxiv.org/abs/2310.15556). The project aims to implement and evaluate a token compression technique for large language models (LLMs), focusing on Retrieval Augmented Generation (RAG) systems which will allow reducing the operation costs when using a third-party LLM that charges by a per-token rate.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd token-compression-llm
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required models:
   - Download a LlamaCPP model (e.g., Llama-2-7B-GGUF) for summarization
   - Download Mistral 7B instruct model for data generation
   - Update the model paths in the scripts or use command line arguments

4. Set up OpenAI API key (required for evaluation):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

1. **Generate summarized dataset:**
```bash
python generate_data.py --dataset_name rag-datasets/mini-bioasq --percent 50 --output_file train50.csv
```

2. **Fine-tune T5 model:**
```bash
python finetune_model.py --mode train --train_data train50.csv --model_name t5-small
```

3. **Create vector index:**
```bash
python rag.py --create_index --dataset_name rag-datasets/mini-bioasq
```

4. **Evaluate RAG system:**
```bash
python rag.py --custom_rag --summarizer_model my_fine_tuned_t5_small_model
```

## Scripts Overview

### `generate_data.py`

This script generates summarized versions of datasets for fine-tuning summarization models. It provides functionality to summarize datasets and export the results to CSV files. Users can specify the percentage of summarization and choose between summarizing the entire dataset or a single entry for testing model output.

**Key Features:**
- Configurable summarization percentage
- Support for different datasets
- Progress tracking and logging
- Error handling and validation
- Single entry testing mode

**Command Line Arguments:**
- `--dataset_name`: Name of the dataset (default: "rag-datasets/mini-bioasq")
- `--percent`: Percentage of summarization (default: 50)
- `--output_file`: Output CSV file name (default: "train.csv")
- `--single_entry`: Test mode - summarize only a single entry
- `--model_path`: Path to the LlamaCPP model file

**Functions:**
- `summarize_dataset()`: Loads dataset, applies summarization model, and generates summaries
- `summarize_entry()`: Summarizes a single text entry for testing
- `setup_logging()`: Configures logging for the application

**Example Usage:**

* **Summarize entire dataset:**
```bash
python generate_data.py --dataset_name rag-datasets/mini-bioasq --percent 50 --output_file train50.csv --model_path ./mistral-7b-instruct-v0.2.Q2_K.gguf
```

* **Test single entry:**
```bash
python generate_data.py --dataset_name rag-datasets/mini-bioasq --single_entry --percent 50
```

* **Generate different compression levels:**
```bash
# 30% compression
python generate_data.py --percent 30 --output_file train30.csv

# 70% compression  
python generate_data.py --percent 70 --output_file train70.csv
```



### `finetune_model.py`

This script fine-tunes T5 summarization models for text compression. It loads training data, preprocesses it, and trains the model with specified hyperparameters. The script includes comprehensive logging, error handling, and evaluation metrics like ROUGE scores.

**Key Features:**
- Configurable model selection (t5-small, t5-base, etc.)
- Comprehensive logging and progress tracking
- Error handling and validation
- Test mode for model evaluation
- Configurable training parameters

**Command Line Arguments:**
- `--mode`: Operation mode - "train" or "test" (required)
- `--train_data`: Path to training CSV file (default: "./train.csv")
- `--model_name`: HuggingFace model name (default: "t5-small")
- `--output_dir`: Output directory for fine-tuned model (default: "my_fine_tuned_t5_small_model")
- `--num_test_samples`: Number of test samples to evaluate (default: 5)

**Usage Examples:**

* **Train a model:**
```bash
python finetune_model.py --mode train --train_data train50.csv --model_name t5-small --output_dir my_t5_model
```

* **Test a trained model:**
```bash
python finetune_model.py --mode test --train_data train50.csv --num_test_samples 10
```

**Components:**

* **Data Processing:**
    - Loads data from CSV files with validation
    - Splits data into train/test sets (80/20 split)
    - Tokenizes data for T5 model training
    - Handles text preprocessing and formatting

* **Model Configuration:**
    - Uses HuggingFace Transformers for T5 model loading
    - Configurable training parameters (learning rate, batch size, epochs)
    - Support for different T5 model sizes
    - Automatic mixed precision training (FP16)

* **Training Process:**
    - Comprehensive logging and progress tracking
    - Evaluation during training with ROUGE metrics
    - Model checkpointing and saving
    - Error handling and recovery

* **Evaluation:**
    - ROUGE score calculation for summarization quality
    - Multiple test samples evaluation
    - Comparison with ground truth summaries
    - Detailed logging of results

**Output:**
- Trained model saved to specified directory
- Training logs saved to `training.log`
- Evaluation results with detailed metrics


### `rag.py`

This script implements and evaluates the Retrieval Augmented Generation (RAG) system with token compression. It includes functions to load vector indices, configure LLMs, set up custom RAG query engines with summarization, and evaluate the system using comprehensive benchmarks.

**Key Features:**
- Support for both standard and compressed RAG systems
- Comprehensive evaluation metrics
- Configurable model paths and settings
- Detailed logging and error handling
- Support for different embedding models

**Command Line Arguments:**
- `--custom_rag`: Use RAG with compressed context (summarization)
- `--dataset_name`: Dataset name (default: "rag-datasets/mini-bioasq")
- `--create_index`: Create new vector index from dataset
- `--index_path`: Path to vector index (default: "full_index")
- `--model_url`: URL to download LLM model
- `--model_path`: Local path to LLM model file
- `--summarizer_model`: Path to summarization model
- `--embed_model`: Embedding model (default: "local:BAAI/bge-small-en-v1.5")
- `--openai_api_key`: OpenAI API key for evaluation 

**System Components:**

1. **Vector Index Management**
   - Creates searchable vector indices from text corpora
   - Supports multiple embedding models
   - Persistent storage for efficient retrieval
   - Automatic index creation and loading

2. **LLM Configuration**
   - Support for various LLM models (Llama, Mistral, etc.)
   - Configurable model parameters (temperature, context window)
   - Local and remote model support
   - Automatic model validation

3. **RAG Query Engine**
   - **Standard RAG**: Uses full context for retrieval
   - **Compressed RAG**: Uses summarized context for reduced token usage
   - Custom query processing with error handling
   - Configurable retrieval and synthesis parameters

4. **Evaluation Framework**
   - Comprehensive metrics using tonic-validate
   - Support for multiple evaluation datasets
   - Detailed performance analysis
   - Comparison between compressed and standard RAG

**Evaluation Metrics:**

1. **Answer Similarity Score (0-5)**
   - Measures semantic similarity between reference and generated answers
   - Uses advanced language models for comparison
   - Evaluates overall answer quality

2. **Retrieval Precision (0-1)**
   - Measures relevance of retrieved context to the question
   - Formula: Relevant retrieved context / Total retrieved context
   - Evaluates retrieval system effectiveness

3. **Augmentation Accuracy (0-1)**
   - Measures how well retrieved context is incorporated into answers
   - Formula: Context used in answer / Total retrieved context
   - Evaluates context utilization

4. **Answer Consistency (0-1)**
   - Measures alignment between retrieved context and generated answers
   - Evaluates factual consistency and grounding
   - Ensures answers are supported by retrieved information


### `test_rag.py`

Interactive testing script for the RAG system. Provides a command-line interface where users can input questions and receive responses from either the compressed RAG (with summarized context) or the standard RAG system. Useful for manual testing and comparison of different configurations. 

## Complete Workflow

### Step 1: Generate Summarized Dataset
```bash
# Generate 50% compressed dataset
python generate_data.py --dataset_name rag-datasets/mini-bioasq --percent 50 --output_file train50.csv --model_path ./mistral-7b-instruct-v0.2.Q2_K.gguf

# Generate 30% compressed dataset
python generate_data.py --percent 30 --output_file train30.csv

# Test single entry
python generate_data.py --single_entry --percent 50
```

### Step 2: Fine-tune T5 Summarization Model
```bash
# Train T5 model on compressed dataset
python finetune_model.py --mode train --train_data train50.csv --model_name t5-small --output_dir my_t5_50_model

# Test the trained model
python finetune_model.py --mode test --train_data train50.csv --num_test_samples 5
```

### Step 3: Create Vector Index
```bash
# Create index from dataset
python rag.py --create_index --dataset_name rag-datasets/mini-bioasq --index_path bioasq_index
```

### Step 4: Evaluate RAG Systems
```bash
# Evaluate compressed RAG
python rag.py --custom_rag --summarizer_model my_t5_50_model --index_path bioasq_index

# Evaluate standard RAG (for comparison)
python rag.py --index_path bioasq_index

# Compare different compression levels
python rag.py --custom_rag --summarizer_model my_t5_30_model --index_path bioasq_index
```

### Step 5: Interactive Testing (Optional)
```bash
python test_rag.py --custom_rag --summarizer_model my_t5_50_model
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for evaluation metrics (set via `--openai_api_key` or environment variable)

### Model Paths
- Update model paths in scripts or use command line arguments
- Default models: Llama-2-7B, Mistral-7B, T5-small
- Support for local and remote model loading 

## Experimental Results

### Performance Comparison

| Configuration | Answer Similarity (0-5) | Answer Consistency (0-1) | Retrieval Precision (0-1) | Augmentation Accuracy (0-1) | Token Reduction |
|---|---|---|---|---|---|
| **Baseline: No RAG** | 2.0 | - | - | - | 0% |
| **Standard RAG** | 4.5 | 0.92 | 0.95 | 0.86 | 0% |
| **Compressed RAG (Generic Summarizer)** | 3.0 | 0.67 | 0.70 | 0.78 | ~50% |
| **Compressed RAG (Fine-tuned 50%)** | 3.8 | 0.76 | 0.85 | 0.80 | ~50% |
| **Compressed RAG (Fine-tuned 70%)** | 4.3 | 0.90 | 0.88 | 0.83 | ~30% |

### Key Findings

1. **Token Compression Effectiveness**: Fine-tuned summarization models achieve significant token reduction (30-50%) while maintaining reasonable performance.

2. **Quality vs. Compression Trade-off**: Higher compression ratios (50%) show more token savings but lower quality scores, while moderate compression (70%) provides better balance.

3. **Fine-tuning Benefits**: Custom fine-tuned models significantly outperform generic summarization approaches, especially in consistency and precision metrics.

4. **Cost Efficiency**: The compressed RAG system can reduce inference costs by 30-50% while maintaining 80-90% of the original performance.

### Dataset Information
- **Dataset**: rag-datasets/mini-bioasq
- **Test Samples**: 100 question-answer pairs
- **Context Length**: Variable (typically 200-1000 tokens)
- **Evaluation Framework**: tonic-validate with GPT-3.5-turbo

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files exist and paths are correct
   - Check available disk space for model downloads
   - Verify model format compatibility (GGUF for LlamaCPP)

2. **Memory Issues**
   - Reduce batch size in training arguments
   - Use smaller models (t5-small instead of t5-base)
   - Enable gradient checkpointing

3. **API Key Issues**
   - Set OPENAI_API_KEY environment variable
   - Use --openai_api_key command line argument
   - Verify API key has sufficient credits

4. **Dataset Loading Problems**
   - Check internet connection for dataset downloads
   - Verify dataset name and format
   - Ensure sufficient disk space

### Performance Optimization

- Use GPU acceleration when available
- Enable mixed precision training (FP16)
- Adjust context window size based on available memory
- Use smaller embedding models for faster indexing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{tcra_llm_2023,
  title={TCRA-LLM: Token Compression Retrieval Augmented Large Language Model for Inference Cost Reduction},
  author={[Authors]},
  journal={arXiv preprint arXiv:2310.15556},
  year={2023}
}
```
