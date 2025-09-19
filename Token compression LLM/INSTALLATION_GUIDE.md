# Installation Guide for Token Compression LLM

## Current Status

✅ **Successfully Installed:**
- Core ML libraries (torch, transformers, datasets, evaluate, accelerate)
- LlamaIndex components (core, embeddings, vector stores, evaluation)
- Data processing libraries (pandas, numpy, nltk)
- Evaluation frameworks (tonic-validate, rouge-score)
- Vector search (faiss-cpu)
- Utilities and other dependencies

❌ **Installation Issue:**
- `llama-cpp-python` - Compilation failed due to missing C++ headers on macOS

## Solutions for llama-cpp-python

### Option 1: Use Conda (Recommended)
```bash
# Install conda if you don't have it
# Then create a new environment
conda create -n token-compression python=3.11
conda activate token-compression
conda install -c conda-forge llama-cpp-python
```

### Option 2: Install Xcode Command Line Tools
```bash
# Install Xcode command line tools
xcode-select --install

# Then try installing again
pip install llama-cpp-python
```

### Option 3: Use Pre-compiled Wheel
```bash
# Try installing from a different source
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Option 4: Alternative LLM Backend
You can modify the scripts to use HuggingFace models instead of LlamaCPP:

```python
# Instead of LlamaCPP, use HuggingFace models
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# For summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# For text generation
generator = pipeline("text-generation", model="gpt2")
```

## Running the Project

### What Works Now:
1. **Data Generation** (`generate_data.py`) - ✅ Ready to use
2. **Model Fine-tuning** (`finetune_model.py`) - ✅ Ready to use  
3. **RAG Evaluation** (`rag.py`) - ⚠️ Needs llama-cpp-python or alternative

### Quick Test:
```bash
# Test the fine-tuning script
python finetune_model.py --mode test --train_data train.csv --num_test_samples 3

# Test data generation (if you have a model file)
python generate_data.py --single_entry --percent 50
```

## Next Steps

1. **Choose one of the solutions above** for llama-cpp-python
2. **Download required models:**
   - T5 model for summarization (will be downloaded automatically)
   - Llama/Mistral model for data generation (if using LlamaCPP)
   - Or use HuggingFace models as alternatives

3. **Set up OpenAI API key** (for evaluation):
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Run the complete workflow:**
   ```bash
   # Generate data
   python generate_data.py --dataset_name rag-datasets/mini-bioasq --percent 50 --output_file train50.csv
   
   # Fine-tune model
   python finetune_model.py --mode train --train_data train50.csv --model_name t5-small
   
   # Create index and evaluate
   python rag.py --create_index --dataset_name rag-datasets/mini-bioasq
   python rag.py --custom_rag --summarizer_model my_fine_tuned_t5_small_model
   ```

## Troubleshooting

### If you continue having issues with llama-cpp-python:
1. Try using HuggingFace models instead
2. Use Google Colab or another environment
3. Consider using Docker with a pre-built image

### For macOS-specific issues:
1. Make sure you have the latest Xcode command line tools
2. Try using conda instead of pip
3. Consider using a virtual environment

## Alternative: Use HuggingFace Models

If llama-cpp-python continues to be problematic, you can modify the scripts to use HuggingFace models instead. The core functionality will work the same way, just with different model backends.

Let me know if you'd like me to help modify the scripts to use HuggingFace models instead!
