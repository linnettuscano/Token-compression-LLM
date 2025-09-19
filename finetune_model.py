import argparse
import pandas as pd
import logging
import os
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
  parser = argparse.ArgumentParser(description="Train or test a T5 summarization model")
  parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                      help="Train or test the model. Test essentially summarizes a single entry and print to console for inspection")
  parser.add_argument("--train_data", type=str, default="./train.csv",
                      help="Path to the training CSV file")
  parser.add_argument("--model_name", type=str, default="t5-small",
                      help="HF Model name")
  parser.add_argument("--output_dir", type=str, default="my_fine_tuned_t5_small_model",
                      help="Output directory for the fine-tuned model")
  parser.add_argument("--num_test_samples", type=int, default=5,
                      help="Number of test samples to evaluate in test mode")

  args = parser.parse_args()
  return args


def preprocess_function(examples):
  inputs = ["summarize: " + doc for doc in examples["original text"]]
  model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
  labels = tokenizer(text_target=examples["summarized text"], max_length=128, truncation=True)
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs


def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
  labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
  decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
  result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
  result["gen_len"] = np.mean(prediction_lens)
  return {k: round(v, 4) for k, v in result.items()}


if __name__ == "__main__":
  args = parse_args()
  logger = setup_logging()
  
  try:
    logger.info(f"Starting {args.mode} mode with model: {args.model_name}")
    
    # Validate input file exists
    if not os.path.exists(args.train_data):
        raise FileNotFoundError(f"Training data file not found: {args.train_data}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    rouge = evaluate.load("rouge")

    if args.mode == "train":
        logger.info("Loading training data...")
        # Load training data
        data = pd.read_csv(args.train_data)
        logger.info(f"Loaded {len(data)} training samples")
        logger.info(f"Data preview:\n{data.head()}")
        
        ds = Dataset.from_pandas(data)
        ds = ds.train_test_split(test_size=0.2)
        logger.info(f"Split data: {len(ds['train'])} train, {len(ds['test'])} test")

        tokenized_data = ds.map(preprocess_function, batched=True)

        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            fp16=True,
            logging_steps=100,
            save_steps=500,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        logger.info("Starting training...")
        trainer.train()
        trainer.save_model(f"my_fine_tuned_{args.model_name}")
        logger.info(f"Training completed. Model saved to: my_fine_tuned_{args.model_name}")

    elif args.mode == "test":
        model_path = f"my_fine_tuned_{args.model_name}"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fine-tuned model not found: {model_path}. Please train the model first.")
        
        logger.info("Loading test data...")
        data = pd.read_csv(args.train_data)
        ds = Dataset.from_pandas(data)
        ds = ds.train_test_split(test_size=0.2)
        
        logger.info(f"Testing on {min(args.num_test_samples, len(ds['test']))} samples")
        
        summarizer = pipeline("summarization", model=model_path)
        
        for i in range(min(args.num_test_samples, len(ds['test']))):
            original_text = ds['test'][i]['original text']
            ground_truth = ds['test'][i]['summarized text']
            
            input_text = "summarize: " + original_text
            logger.info(f"\n--- Sample {i+1} ---")
            logger.info(f"Original text: {original_text[:200]}...")
            logger.info(f"Ground truth: {ground_truth}")
            
            try:
                pred = summarizer(input_text, max_length=128, min_length=30, do_sample=False)
                logger.info(f"Predicted summary: {pred[0]['summary_text']}")
            except Exception as e:
                logger.error(f"Error generating summary for sample {i+1}: {e}")
            
  except Exception as e:
      logger.error(f"An error occurred: {e}")
      raise