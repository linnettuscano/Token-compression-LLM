import argparse
import pandas as pd
import logging
import os
from datasets import load_dataset
from nltk import word_tokenize
from llama_index.core import Settings
from llama_index.llms.llama_cpp import LlamaCPP


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_generation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def summarize_entry(llm, text, percent=50, logger=None):
    question = f"""
    Instruction : Summarize the following text, the length of the summary result is {percent} percent of the original text,
     keep the first sentence, and directly output your answer: \n\n {text}
    """
    resp = llm.complete(question)
    summary = resp.text.replace("Summary: ", "")
    summary_len = len(word_tokenize(summary))
    original_len = len(word_tokenize(text))
    percentage_reduction = (1 - (summary_len / original_len)) * 100

    if logger:
        logger.info("\nSummarization Output on Single Entry:")
        logger.info(summary)
        logger.info(f"Original length: {original_len}")
        logger.info(f"Summarized length: {summary_len}")
        logger.info(f"Percentage reduction: {percentage_reduction:.2f}%")
    else:
        print("\nSummarization Output on Single Entry:")
        print(summary)
        print("OG len : ", original_len)
        print("Summarized len : ", summary_len)
        print("Percentage reduction: ", percentage_reduction)

def summarize_dataset(dataset_name, percent=50, output_file="train.csv", single_entry=False, model_path="./mistral-7b-instruct-v0.2.Q2_K.gguf", logger=None):
    try:
        if logger:
            logger.info(f"Loading dataset: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name, 'question-answer-passages')
        data = load_dataset(dataset_name, 'text-corpus')
        text_list = [x['passage'] for x in data['passages']]
        
        if logger:
            logger.info(f"Loaded {len(text_list)} text passages")

        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if logger:
            logger.info(f"Loading model from: {model_path}")
        
        # Load summarization model
        llm = LlamaCPP(
            model_path=model_path,
            temperature=0.2,
            max_new_tokens=256,
            context_window=3900,
            generate_kwargs={},
            verbose=False,  # Reduce verbosity
        )
        Settings.llm = llm
        Settings.chunk_size = 1024

        if single_entry:
            if logger:
                logger.info("Processing single entry for testing...")
            summarize_entry(llm, text_list[0], percent, logger)
            return

        reduced_text_list = []
        total_texts = len(text_list)
        
        if logger:
            logger.info(f"Starting summarization of {total_texts} texts...")

        for idx, text in enumerate(text_list):
            if logger and idx % 10 == 0:
                logger.info(f"Processing text {idx+1}/{total_texts}")
            
            original_value = len(word_tokenize(text))
            reduced_value = original_value
            ftext = text
            
            if original_value <= 40:
                reduced_text_list.append(ftext)
                continue
                
            cnt = 0
            while cnt < 3 and reduced_value > original_value * (percent/100):
                try:
                    question = f"""
                    Instruction : Summarize the following text, the length of the summary result is {percent} percent of the original text,
                    keep the first sentence, and directly output your answer please make sure the length of summary is around 50% of input length
                    it is very important also don't remove important factual information: \n\n {ftext}
                    """
                    resp = llm.complete(question)
                    summary = resp.text.replace("Summary: ", "")
                    ftext = summary
                    reduced_value = len(word_tokenize(summary))
                    cnt = cnt + 1
                except Exception as e:
                    if logger:
                        logger.error(f"Error processing text {idx+1}: {e}")
                    # Use original text if summarization fails
                    summary = ftext
                    break
                    
            reduced_text_list.append(summary)

        # Save to DataFrame and export to CSV
        if logger:
            logger.info(f"Saving results to {output_file}...")
            
        df = pd.DataFrame({'original text': text_list, 'summarized text': reduced_text_list})
        df = df[~df['original text'].str.contains('nan')]
        df.to_csv(output_file, index=False)
        
        if logger:
            logger.info(f"Successfully saved {len(df)} entries to {output_file}")
            
    except Exception as e:
        if logger:
            logger.error(f"Error in summarize_dataset: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate summarized version of a dataset for fine-tuning summarization models.")
    parser.add_argument("--dataset_name", type=str, default="rag-datasets/mini-bioasq", help="Name of the dataset to be summarized")
    parser.add_argument("--percent", type=int, default=50, help="Percentage of summarization (default: 50)")
    parser.add_argument("--output_file", type=str, default="train.csv", help="Output file name (default: train.csv)")
    parser.add_argument("--single_entry", action="store_true", help="Summarize a single entry from the dataset to test model output")
    parser.add_argument("--model_path", type=str, default="./mistral-7b-instruct-v0.2.Q2_K.gguf", help="Path to the LlamaCPP model file")
    args = parser.parse_args()
    
    # Validate arguments
    if args.percent <= 0 or args.percent > 100:
        raise ValueError("Percent must be between 1 and 100")
    
    logger = setup_logging()
    
    try:
        logger.info(f"Starting data generation with dataset: {args.dataset_name}")
        summarize_dataset(
            args.dataset_name, 
            percent=args.percent, 
            output_file=args.output_file, 
            single_entry=args.single_entry,
            model_path=args.model_path,
            logger=logger
        )
        logger.info("Data generation completed successfully")
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        raise

if __name__ == "__main__":
    main()
