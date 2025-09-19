#!/usr/bin/env python3
"""
Interactive RAG Testing Script
Allows users to test both standard and custom RAG systems interactively
"""

import argparse
import logging
import os
from datasets import load_dataset
from transformers import pipeline
from llama_index.core import (
    Document, 
    VectorStoreIndex, 
    Settings,
    load_index_from_storage,
    StorageContext,
    PromptTemplate,
    get_response_synthesizer
)
from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.llms import MockLLM
from tonic_validate import Benchmark, ValidateScorer
from tonic_validate.metrics import (
    RetrievalPrecisionMetric,
    AnswerConsistencyMetric, 
    AnswerSimilarityMetric,
    AugmentationAccuracyMetric
)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_rag.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_index(index_path, logger):
    """Load vector index from storage"""
    try:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index path not found: {index_path}")
        
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context=storage_context)
        logger.info(f"Successfully loaded index from {index_path}")
        return index
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        raise

def setup_llm(logger=None):
    """Setup LLM for RAG - using MockLLM for testing"""
    try:
        if logger:
            logger.info("Setting up MockLLM for testing")
        
        # Use MockLLM for testing - this will return predictable responses
        llm = MockLLM(max_tokens=256)
        
        if logger:
            logger.info("LLM setup completed successfully")
        return llm
    except Exception as e:
        if logger:
            logger.error(f"Error setting up LLM: {e}")
        raise

def setup_rag_query_engine(index, llm, summarizer_model="facebook/bart-large-cnn", logger=None):
    """Setup custom RAG query engine with summarization"""
    try:
        if logger:
            logger.info(f"Setting up RAG query engine with summarizer: {summarizer_model}")
        
        # Load summarization model
        summarizer = pipeline("summarization", model=summarizer_model)
        
        retriever = index.as_retriever()
        synthesizer = get_response_synthesizer(response_mode="compact")
        
        qa_prompt = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        )

        class RAGStringQueryEngine(CustomQueryEngine):
            retriever: BaseRetriever
            response_synthesizer: BaseSynthesizer
            llm: MockLLM
            qa_prompt: PromptTemplate

            def custom_query(self, query_str: str):
                try:
                    nodes = self.retriever.retrieve(query_str)
                    context = []
                    
                    for n in nodes:
                        content = n.node.get_content()
                        # Summarize each retrieved context
                        if len(content) > 100:  # Only summarize if content is long enough
                            summary = summarizer(content, max_length=100, min_length=30, do_sample=False)
                            context.append(summary[0]['summary_text'])
                        else:
                            context.append(content)
                    
                    context_str = "\n\n".join(context)
                    response = self.llm.complete(
                        self.qa_prompt.format(context_str=context_str, query_str=query_str)
                    )

                    return {
                        "llm_answer": str(response),
                        "llm_context_list": context,
                        "original_context": [n.node.get_content() for n in nodes]
                    }
                except Exception as e:
                    if logger:
                        logger.error(f"Error in custom query: {e}")
                    return {
                        "llm_answer": f"Error processing query: {e}",
                        "llm_context_list": [],
                        "original_context": []
                    }

        query_engine = RAGStringQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt,
        )
        
        if logger:
            logger.info("RAG query engine setup completed")
        return query_engine
    except Exception as e:
        if logger:
            logger.error(f"Error setting up RAG query engine: {e}")
        raise

def setup_scorer(logger=None):
    """Setup evaluation scorer"""
    try:
        scorer = ValidateScorer([
            AnswerSimilarityMetric(), 
            AnswerConsistencyMetric(),
            RetrievalPrecisionMetric(),
            AugmentationAccuracyMetric()
        ])
        if logger:
            logger.info("Evaluation scorer setup completed")
        return scorer
    except Exception as e:
        if logger:
            logger.error(f"Error setting up scorer: {e}")
        raise

def get_llama_response(query, query_engine, logger=None):
    """Get response from standard query engine"""
    try:
        response = query_engine.query(query)
        context = [x.text for x in response.source_nodes]
        return {
            "llm_answer": response.response,
            "llm_context_list": context
        }
    except Exception as e:
        if logger:
            logger.error(f"Error getting response: {e}")
        return {
            "llm_answer": f"Error: {e}",
            "llm_context_list": []
        }

def interactive_test(query_engine, custom_rag=False, logger=None):
    """Interactive testing loop"""
    if logger:
        logger.info("Starting interactive test mode")
    
    print("\n" + "="*60)
    print("🤖 Interactive RAG Testing")
    print("="*60)
    print("Enter your questions (type 'quit' to exit)")
    print("-"*60)
    
    while True:
        try:
            question = input("\n❓ Enter question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not question:
                print("⚠️  Please enter a valid question")
                continue
            
            print(f"\n🔍 Processing: {question}")
            print("-"*40)
            
            if custom_rag:
                result = query_engine.custom_query(question)
                print(f"📝 Answer: {result['llm_answer']}")
                print(f"\n📚 Summarized Context ({len(result['llm_context_list'])} items):")
                for i, ctx in enumerate(result['llm_context_list'], 1):
                    print(f"  {i}. {ctx[:100]}{'...' if len(ctx) > 100 else ''}")
            else:
                result = get_llama_response(question, query_engine, logger)
                print(f"📝 Answer: {result['llm_answer']}")
                print(f"\n📚 Context ({len(result['llm_context_list'])} items):")
                for i, ctx in enumerate(result['llm_context_list'], 1):
                    print(f"  {i}. {ctx[:100]}{'...' if len(ctx) > 100 else ''}")
            
            print("-"*40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            if logger:
                logger.error(f"Error in interactive test: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Interactive RAG Testing')
    parser.add_argument('--custom_rag', action='store_true', help='Use custom RAG with summarization')
    parser.add_argument('--create_index', action='store_true', help='Create new vector index from dataset')
    parser.add_argument("--dataset_name", type=str, default="rag-datasets/mini-bioasq", 
                       help="Name of the dataset to use")
    parser.add_argument("--index_path", type=str, default="full_index", 
                       help="Path to the vector index directory")
    parser.add_argument("--summarizer_model", type=str, default="facebook/bart-large-cnn", 
                       help="HuggingFace model for summarization")
    parser.add_argument("--embed_model", type=str, default="local:BAAI/bge-small-en-v1.5", 
                       help="Embedding model to use")
    args = parser.parse_args()

    logger = setup_logging()
    
    try:
        logger.info("Starting interactive RAG testing")
        
        # Setup embedding model
        Settings.embed_model = resolve_embed_model(args.embed_model)
        logger.info(f"Using embedding model: {args.embed_model}")
        
        # Load or create index
        index = None
        if args.create_index:
            logger.info("Creating new index...")
            data = load_dataset(args.dataset_name, 'text-corpus')
            text_list = [x['passage'] for x in data['passages']]
            documents = [Document(text=t) for t in text_list]
            index = VectorStoreIndex.from_documents(documents)
            index.storage_context.persist(args.index_path)
            logger.info(f"Index created and saved to: {args.index_path}")
        else:
            index = load_index(args.index_path, logger)

        # Setup LLM
        llm = setup_llm(logger)
        Settings.llm = llm

        # Setup query engine
        if args.custom_rag:
            logger.info("Using custom RAG with summarization")
            query_engine = setup_rag_query_engine(index, llm, args.summarizer_model, logger)
        else:
            logger.info("Using standard RAG")
            query_engine = index.as_query_engine()

        # Start interactive testing
        interactive_test(query_engine, args.custom_rag, logger)
        
        logger.info("Interactive testing completed")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        print("Please check the logs for more details")

if __name__ == "__main__":
    main()