#!/usr/bin/env python3
"""
Test script to verify installation and basic functionality
"""

import sys
import logging

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        # Core ML libraries
        import torch
        import transformers
        import datasets
        import pandas as pd
        import numpy as np
        print("✅ Core ML libraries: OK")
        
        # LlamaIndex components
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print("✅ LlamaIndex components: OK")
        
        # Evaluation framework
        import tonic_validate
        from tonic_validate import Benchmark, ValidateScorer
        print("✅ Evaluation framework: OK")
        
        # Data processing
        import nltk
        from rouge_score import rouge_scorer
        print("✅ Data processing: OK")
        
        # Vector search
        import faiss
        print("✅ Vector search: OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without llama-cpp-python"""
    print("\nTesting basic functionality...")
    
    try:
        # Test HuggingFace model loading
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
        print("✅ T5 model loading: OK")
        
        # Test dataset loading
        from datasets import load_dataset
        dataset = load_dataset("squad", split="train[:10]")  # Small sample
        print("✅ Dataset loading: OK")
        
        # Test LlamaIndex document creation
        from llama_index.core import Document
        doc = Document(text="This is a test document.")
        print("✅ Document creation: OK")
        
        # Test embedding model
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        print("✅ Embedding model: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

def test_llama_cpp_alternative():
    """Test if we can work around llama-cpp-python"""
    print("\nTesting alternatives to llama-cpp-python...")
    
    try:
        # Test HuggingFace pipeline for summarization
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("✅ HuggingFace summarization: OK")
        
        # Test text generation
        generator = pipeline("text-generation", model="gpt2")
        print("✅ HuggingFace text generation: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Alternative error: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Token Compression LLM - Installation Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        # Test alternatives
        alternatives_ok = test_llama_cpp_alternative()
        
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print("=" * 50)
        
        if imports_ok and functionality_ok:
            print("✅ Core installation: SUCCESS")
            print("✅ You can run most of the project!")
            
            if alternatives_ok:
                print("✅ Alternative models: AVAILABLE")
                print("✅ You can run the complete workflow!")
            else:
                print("⚠️  Alternative models: ISSUES")
                print("⚠️  You may need to fix llama-cpp-python")
                
        else:
            print("❌ Installation: ISSUES")
            print("❌ Please check the error messages above")
            
    else:
        print("❌ Basic imports failed - please check installation")
    
    print("\nNext steps:")
    print("1. If everything is OK, you can start using the project")
    print("2. If llama-cpp-python is missing, see INSTALLATION_GUIDE.md")
    print("3. Run: python finetune_model.py --mode test --train_data train.csv")

if __name__ == "__main__":
    main()
