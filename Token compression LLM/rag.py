import argparse
import logging
import os
import json
from datasets import load_dataset
from transformers import pipeline
from llama_index.core import (
    Document, 
    VectorStoreIndex, 
    SimpleDirectoryReader, 
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
from llama_index.llms.llama_cpp import LlamaCPP
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
            logging.FileHandler('rag_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_index(index_path, logger=None):
    """Load vector index from storage"""
    try:
        if logger:
            logger.info(f"Loading index from: {index_path}")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index path does not exist: {index_path}")
            
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context=storage_context)
        
        if logger:
            logger.info("Index loaded successfully")
        
        return index
    except Exception as e:
        if logger:
            logger.error(f"Error loading index: {e}")
        raise

def setup_llm(model_url=None, model_path=None, logger=None):
    """Setup LLM for RAG"""
    try:
        if logger:
            logger.info("Setting up LLM...")
        
        # Default model if none provided
        if not model_url and not model_path:
            model_url = "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf"
        
        # Validate model path if provided
        if model_path and not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        llm = LlamaCPP(
            model_url=model_url,
            model_path=model_path,
            temperature=0.1,
            max_new_tokens=256,
            context_window=3900,
            generate_kwargs={},
            verbose=False,  # Reduce verbosity
        )
        
        if logger:
            logger.info("LLM setup completed successfully")
        
        return llm
    except Exception as e:
        if logger:
            logger.error(f"Error setting up LLM: {e}")
        raise

def setup_rag_query_engine(index, llm, summarizer_model_path="my_fine_tuned_t5_small_model", logger=None):
    """Setup custom RAG query engine with summarization"""
    try:
        if logger:
            logger.info(f"Setting up RAG query engine with summarizer: {summarizer_model_path}")
        
        # Validate summarizer model path
        if not os.path.exists(summarizer_model_path):
            raise FileNotFoundError(f"Summarizer model not found: {summarizer_model_path}")
        
        summarizer = pipeline("summarization", model=summarizer_model_path)
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
            llm: LlamaCPP 
            qa_prompt: PromptTemplate
            summarizer: pipeline
            logger: logging.Logger

            def custom_query(self, query_str: str):
                try:
                    nodes = self.retriever.retrieve(query_str)
                    context = []
                    for n in nodes:
                        try:
                            summary = self.summarizer(n.node.get_content())[0]['summary_text']
                            context.append(summary)
                        except Exception as e:
                            if self.logger:
                                self.logger.warning(f"Failed to summarize node: {e}")
                            context.append(n.node.get_content())
                    
                    context_str = "\n\n".join(context)
                    response = self.llm.complete(
                        self.qa_prompt.format(context_str=context_str, query_str=query_str)
                    )

                    return {"llm_answer": response, "llm_context_list": context}
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Error in custom_query: {e}")
                    raise
            
        query_engine = RAGStringQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            llm=llm,
            qa_prompt=qa_prompt,
            summarizer=summarizer,
            logger=logger
        )
        
        if logger:
            logger.info("RAG query engine setup completed successfully")
        
        return query_engine
    except Exception as e:
        if logger:
            logger.error(f"Error setting up RAG query engine: {e}")
        raise

def setup_scorer(logger=None):
    """Setup validation scorer"""
    try:
        if logger:
            logger.info("Setting up validation scorer...")
        
        scorer = ValidateScorer([
            AnswerSimilarityMetric(), 
            AnswerConsistencyMetric(),
            RetrievalPrecisionMetric(),
            AugmentationAccuracyMetric()
        ])
        
        if logger:
            logger.info("Validation scorer setup completed successfully")
        
        return scorer
    except Exception as e:
        if logger:
            logger.error(f"Error setting up scorer: {e}")
        raise

def get_llama_response(prompt, query_engine, logger=None):
    """Get response from standard LlamaIndex query engine"""
    try:
        response = query_engine.query(prompt)
        context = [x.text for x in response.source_nodes]
        return {
            "llm_answer": response.response,
            "llm_context_list": context
        }
    except Exception as e:
        if logger:
            logger.error(f"Error getting Llama response: {e}")
        raise

def get_custom_response(prompt, query_engine, logger=None):
    """Get response from custom RAG query engine"""
    try:
        response = query_engine.custom_query(prompt)
        return {
            "llm_answer": response["llm_answer"],
            "llm_context_list": response["llm_context_list"]
        }
    except Exception as e:
        if logger:
            logger.error(f"Error getting custom response: {e}")
        raise


def evaluate(benchmark, query_fn, scorer, logger=None):
    """Evaluate RAG system using benchmark"""
    try:
        if logger:
            logger.info("Starting evaluation...")
        
        run = scorer.score(benchmark, query_fn)
        
        if logger:
            logger.info("Overall Scores")
            logger.info(run.overall_scores)
            logger.info("------")
            
            for i, item in enumerate(run.run_data):
                logger.info(f"Question {i+1}: {item.reference_question}")
                logger.info(f"Answer: {item.reference_answer}")
                logger.info(f"LLM Answer: {item.llm_answer}")
                logger.info(f"LLM Context: {item.llm_context}")
                logger.info(f"Scores: {item.scores}")
                logger.info("------")
        else:
            print("Overall Scores")
            print(run.overall_scores)
            print("------")
            for item in run.run_data:
                print("Question: ", item.reference_question)
                print("Answer: ", item.reference_answer)
                print("LLM Answer: ", item.llm_answer)
                print("LLM Context: ", item.llm_context)
                print("Scores: ", item.scores)
                print("------")
        
        if logger:
            logger.info("Evaluation completed successfully")
            
    except Exception as e:
        if logger:
            logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAG Evaluation')
    parser.add_argument('--custom_rag', action='store_true', help='Use custom RAG')
    parser.add_argument("--dataset_name", type=str, default="rag-datasets/mini-bioasq", help="Name of the dataset used (make sure it is a QA dataset with both QA and text corpus)")
    parser.add_argument('--create_index', action='store_true', help='Download dataset and create index')
    parser.add_argument('--index_path', type=str, default="full_index", help="Path to the vector index")
    parser.add_argument('--model_url', type=str, help="URL to download LLM model")
    parser.add_argument('--model_path', type=str, help="Local path to LLM model file")
    parser.add_argument('--summarizer_model', type=str, default="my_fine_tuned_t5_small_model", help="Path to the summarization model")
    parser.add_argument('--embed_model', type=str, default="local:BAAI/bge-small-en-v1.5", help="Embedding model to use")
    parser.add_argument('--openai_api_key', type=str, help="OpenAI API key for tonic_validate (can also set OPENAI_API_KEY env var)")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Setup OpenAI API key
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    elif not os.environ.get("OPENAI_API_KEY"):
        logger.warning("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or use --openai_api_key argument")
    
    try:
        logger.info(f"Starting RAG evaluation with dataset: {args.dataset_name}")
        
        # Setup embedding model
        Settings.embed_model = resolve_embed_model(args.embed_model)
        logger.info(f"Using embedding model: {args.embed_model}")
        
        # Load dataset
        dataset = load_dataset(args.dataset_name, 'question-answer-passages')
        logger.info(f"Loaded dataset with {len(dataset['test'])} test samples")
        
        # Create or load index
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
        llm = setup_llm(args.model_url, args.model_path, logger)
        Settings.llm = llm
        
        # Setup scorer
        scorer = setup_scorer(logger)

        # Prepare evaluation data
        eval_questions = [x for x in dataset['test']['question']]
        eval_answers = [x for x in dataset['test']['answer']]
        benchmark = Benchmark(questions=eval_questions, answers=eval_answers)
        logger.info(f"Prepared benchmark with {len(eval_questions)} questions")
        
        # Run evaluation
        if args.custom_rag:
            logger.info("Using custom RAG with summarization")
            query_engine = setup_rag_query_engine(index, llm, args.summarizer_model, logger)
            evaluate(benchmark, query_engine.custom_query, scorer, logger)
        else:
            logger.info("Using standard LlamaIndex query engine")
            query_engine = index.as_query_engine()
            evaluate(benchmark, lambda prompt: get_llama_response(prompt, query_engine, logger), scorer, logger)
        
        logger.info("RAG evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"RAG evaluation failed: {e}")
        raise
