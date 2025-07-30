"""
Domain Name Model Module

This module contains the DomainNameModel class for fine-tuning and inference
of domain name suggestion models using LoRA (Low-Rank Adaptation).
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from datasets import Dataset
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import re
import numpy as np

logger = logging.getLogger(__name__)


class DomainNameModel:
    """
    A fine-tuned language model for generating domain name suggestions
    based on business descriptions.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the DomainNameModel.
        
        Args:
            config_path: Path to the model configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Model components
        self.tokenizer = None
        self.base_model = None
        self.model = None
        self.peft_config = None
        self.trainer = None
        
        # Training state
        self.is_trained = False
        self.model_path = None
        
        logger.info(f"Initialized DomainNameModel with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_base_model(self):
        """Load the base model and tokenizer."""
        model_name = self.config['base_model']['name']
        
        logger.info(f"Loading base model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Setup LoRA configuration
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['fine_tuning']['lora']['r'],
            lora_alpha=self.config['fine_tuning']['lora']['lora_alpha'],
            lora_dropout=self.config['fine_tuning']['lora']['lora_dropout'],
            target_modules=self.config['fine_tuning']['lora']['target_modules']
        )
        
        # Get PEFT model
        self.model = get_peft_model(self.base_model, self.peft_config)
        self.model.print_trainable_parameters()
        
    
    def prepare_dataset(self, data: List[Dict[str, Any]]) -> Dataset:
        """
        Prepare dataset for training/evaluation.
        
        Args:
            data: List of data samples with business_description and domain_suggestions
            
        Returns:
            Dataset ready for training
        """
        def format_prompt(business_desc: str, domain_suggestions: List[str] = None) -> str:
            """Format the training prompt."""
            prompt = f"Business Description: {business_desc}\n\nSuggested Domain Names:"
            
            if domain_suggestions:
                for i, domain in enumerate(domain_suggestions[:5], 1):
                    prompt += f"\n{i}. {domain}"
            
            return prompt
        
        # Format the data
        formatted_data = []
        for sample in data:
            text = format_prompt(
                sample['business_description'],
                sample.get('domain_suggestions', [])
            )
            formatted_data.append({"text": text})
        
        # Tokenize the data
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config['fine_tuning']['max_length'],
                return_tensors="pt"
            )
            
            # For language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info(f"Prepared dataset with {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset, output_dir: Path) -> Dict[str, Any]:
        """
        Train the model using the provided datasets.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset  
            output_dir: Directory to save the trained model
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting model training...")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config['fine_tuning']['training']['num_epochs'],
            per_device_train_batch_size=self.config['fine_tuning']['training']['batch_size'],
            per_device_eval_batch_size=self.config['fine_tuning']['training']['eval_batch_size'],
            warmup_steps=self.config['fine_tuning']['training']['warmup_steps'],
            logging_steps=self.config['fine_tuning']['training']['logging_steps'],
            evaluation_strategy="steps",
            eval_steps=self.config['fine_tuning']['training']['eval_steps'],
            save_steps=self.config['fine_tuning']['training']['save_steps'],
            learning_rate=self.config['fine_tuning']['training']['learning_rate'],
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=self.config['fine_tuning']['training']['gradient_accumulation_steps'],
            dataloader_drop_last=True,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for now
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Mark as trained
        self.is_trained = True
        self.model_path = output_dir
        
        logger.info(f"Training completed. Model saved to: {output_dir}")
        
        return {
            "output_dir": str(output_dir),
            "train_loss": train_result.training_loss,
            "eval_loss": self.trainer.state.best_metric if hasattr(self.trainer.state, 'best_metric') else None,
            "global_step": train_result.global_step,
        }
    
    def save_model(self, save_path: Path):
        """Save the trained model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.model and self.tokenizer:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model saved to: {save_path}")
        else:
            logger.warning("No model to save")
    
    def load_trained_model(self, model_path: Path):
        """Load a previously trained model."""
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        logger.info(f"Loading trained model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load the base model first
        base_model_name = self.config['base_model']['name']
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Load the PEFT model
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.is_trained = True
        self.model_path = model_path
        
        logger.info("Trained model loaded successfully")
    
    def generate_domain_suggestions(
        self, 
        business_description: str, 
        num_suggestions: int = 5,
        temperature: float = 0.7,
        max_new_tokens: int = 100
    ) -> List[str]:
        """
        Generate domain name suggestions for a given business description.
        
        Args:
            business_description: Description of the business
            num_suggestions: Number of suggestions to generate
            temperature: Sampling temperature
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            List of domain name suggestions
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_base_model() or load_trained_model() first.")
        
        # Format the prompt
        prompt = f"Business Description: {business_description}\n\nSuggested Domain Names:"
        
        # Tokenize the input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Set model to eval mode
        self.model.eval()
        
        # Generate
        with torch.no_grad():
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
            
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse domain suggestions from generated text
        suggestions = self._parse_domain_suggestions(generated_text, num_suggestions)
        
        return suggestions
    
    def _parse_domain_suggestions(self, generated_text: str, num_suggestions: int) -> List[str]:
        """
        Parse domain suggestions from generated text.
        
        Args:
            generated_text: The generated text containing domain suggestions
            num_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of parsed domain suggestions
        """
        suggestions = []
        
        # Split by lines and look for domain patterns
        lines = generated_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove numbering (1. 2. etc.)
            line = re.sub(r'^[\d\.\-\*\+]*\s*', '', line)
            
            # Look for domain-like patterns
            domain_pattern = r'([a-zA-Z0-9][a-zA-Z0-9\-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,})'
            matches = re.findall(domain_pattern, line.lower())
            
            if matches:
                for match in matches:
                    if match not in suggestions and len(suggestions) < num_suggestions:
                        suggestions.append(match)
            elif line and len(line.split()) <= 3:  # Likely a domain name without TLD
                # Add .com if no TLD present
                if '.' not in line:
                    line = line + '.com'
                if line not in suggestions and len(suggestions) < num_suggestions:
                    suggestions.append(line.lower())
        
        # If we don't have enough suggestions, generate some fallbacks
        if len(suggestions) < num_suggestions:
            # Simple fallback generation (this could be improved)
            base_suggestions = [
                "businessname.com",
                "mycompany.net", 
                "newbusiness.org",
                "startup.io",
                "company.co"
            ]
            
            for suggestion in base_suggestions:
                if len(suggestions) >= num_suggestions:
                    break
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
        
        return suggestions[:num_suggestions]
    
    def batch_generate(
        self, 
        business_descriptions: List[str],
        num_suggestions: int = 5,
        batch_size: int = 4
    ) -> List[List[str]]:
        """
        Generate domain suggestions for multiple business descriptions.
        
        Args:
            business_descriptions: List of business descriptions
            num_suggestions: Number of suggestions per description
            batch_size: Batch size for processing
            
        Returns:
            List of lists, each containing domain suggestions
        """
        all_suggestions = []
        
        for i in range(0, len(business_descriptions), batch_size):
            batch = business_descriptions[i:i + batch_size]
            batch_results = []
            
            for desc in batch:
                suggestions = self.generate_domain_suggestions(
                    desc, 
                    num_suggestions=num_suggestions
                )
                batch_results.append(suggestions)
            
            all_suggestions.extend(batch_results)
        
        return all_suggestions
        
        logger.info(f"Model saved to {save_path}")
        
    def load_model(self, model_path: str):
        """Load a saved model."""
        model_path = Path(model_path)
        
        # Load metadata
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                logger.info(f"Loading model version: {metadata.get('version', 'unknown')}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        
        logger.info(f"Model loaded from {model_path}")
        
    def evaluate_on_dataset(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Please train the model first.")
        
        eval_results = self.trainer.evaluate(eval_dataset=eval_dataset)
        return eval_results

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = DomainNameModel()
    
    # Load base model
    model.load_base_model()
    
    # Setup LoRA
    model.setup_lora()
    
    # Example generation (requires trained model)
    try:
        suggestions = model.generate_suggestions("organic coffee shop in downtown")
        print("Generated suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    except ValueError as e:
        print(f"Cannot generate suggestions: {e}")
        print("Please train the model first!")
