"""
Real Domain Name Model Implementation
Fine-tunes actual LLMs using LoRA and other techniques
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    EarlyStoppingCallback, set_seed
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import yaml
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np
from datetime import datetime
import wandb

logger = logging.getLogger(__name__)

class RealDomainNameModel:
    """
    Real implementation of domain name model with actual fine-tuning
    """
    
    def __init__(self, config_path: str):
        """Initialize with configuration"""
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
        self.training_history = []
        
        # Set random seed for reproducibility
        set_seed(42)
        
        logger.info(f"Initialized RealDomainNameModel with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def load_base_model(self):
        """Load the base model and tokenizer"""
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
        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        logger.info(f"✅ Base model loaded: {model_name}")
        logger.info(f"Model parameters: {self.base_model.num_parameters():,}")
        
        return True
    
    def setup_lora(self):
        """Setup LoRA configuration for efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config['fine_tuning']['lora']['r'],
            lora_alpha=self.config['fine_tuning']['lora']['lora_alpha'],
            lora_dropout=self.config['fine_tuning']['lora']['lora_dropout'],
            target_modules=self.config['fine_tuning']['lora']['target_modules'],
            bias=self.config['fine_tuning']['lora']['bias']
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, lora_config)
        self.peft_config = lora_config
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"All parameters: {all_params:,}")
        logger.info(f"Trainable %: {100 * trainable_params / all_params:.2f}%")
        
        return True
    
    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """Prepare dataset for training"""
        logger.info(f"Preparing dataset with {len(data)} samples")
        
        def format_prompt(example):
            """Format training prompt"""
            business_desc = example['business_description']
            domain_suggestions = example['domain_suggestions']
            
            # Create instruction-following format
            prompt = f"Generate 3 domain name suggestions for this business:\n\nBusiness: {business_desc}\n\nDomain suggestions:\n"
            
            # Add domain suggestions
            domains = []
            for i, domain in enumerate(domain_suggestions[:3], 1):
                domains.append(f"{i}. {domain}")
            
            target = "\n".join(domains)
            full_text = prompt + target + self.tokenizer.eos_token
            
            return full_text
        
        # Format all examples
        formatted_texts = []
        for example in data:
            formatted_texts.append(format_prompt(example))
        
        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_texts})
        
        # Tokenize
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.config['data']['max_input_length'],
                return_tensors=None
            )
            # Add labels for language modeling
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"✅ Dataset prepared: {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset, 
              output_dir: str, experiment_name: str = "baseline") -> Dict[str, Any]:
        """Train the model with real fine-tuning"""
        
        # Setup LoRA
        self.setup_lora()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if available
        if os.getenv('WANDB_API_KEY'):
            wandb.init(
                project="domain-name-llm",
                name=experiment_name,
                config=self.config
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            overwrite_output_dir=True,
            num_train_epochs=self.config['fine_tuning']['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['fine_tuning']['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['fine_tuning']['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['fine_tuning']['training']['gradient_accumulation_steps'],
            learning_rate=self.config['fine_tuning']['training']['learning_rate'],
            weight_decay=self.config['fine_tuning']['training']['weight_decay'],
            logging_steps=self.config['fine_tuning']['training']['logging_steps'],
            eval_steps=self.config['fine_tuning']['training']['eval_steps'],
            save_steps=self.config['fine_tuning']['training']['save_steps'],
            evaluation_strategy=self.config['fine_tuning']['training']['evaluation_strategy'],
            save_strategy="steps",
            load_best_model_at_end=self.config['fine_tuning']['training']['load_best_model_at_end'],
            metric_for_best_model=self.config['fine_tuning']['training']['metric_for_best_model'],
            greater_is_better=self.config['fine_tuning']['training']['greater_is_better'],
            warmup_steps=self.config['fine_tuning']['training']['warmup_steps'],
            fp16=self.config['fine_tuning']['training']['fp16'] and torch.cuda.is_available(),
            dataloader_num_workers=self.config['fine_tuning']['training']['dataloader_num_workers'],
            remove_unused_columns=self.config['fine_tuning']['training']['remove_unused_columns'],
            push_to_hub=False,
            report_to=["wandb"] if os.getenv('WANDB_API_KEY') else [],
            logging_dir=str(output_path / "logs"),
            save_total_limit=3,
            seed=42,
            data_seed=42,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        logger.info("🚀 Starting model training...")
        start_time = datetime.now()
        
        try:
            train_result = self.trainer.train()
            
            # Save training results
            self.training_history = self.trainer.state.log_history
            self.is_trained = True
            self.model_path = output_path
            
            # Save model and tokenizer
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_path)
            
            # Save training metrics
            training_duration = (datetime.now() - start_time).total_seconds()
            
            results = {
                'model_name': experiment_name,
                'training_duration_seconds': training_duration,
                'final_train_loss': train_result.training_loss,
                'final_eval_loss': self.trainer.evaluate()['eval_loss'],
                'total_steps': train_result.global_step,
                'epochs_completed': train_result.epoch,
                'best_model_checkpoint': self.trainer.state.best_model_checkpoint,
                'training_history': self.training_history[-10:],  # Last 10 entries
                'model_path': str(output_path),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save results to JSON
            with open(output_path / 'training_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Training completed in {training_duration:.2f} seconds")
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            logger.info(f"Final eval loss: {results['final_eval_loss']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def generate_domain_suggestions(self, business_description: str, 
                                   num_suggestions: int = 3, 
                                   max_length: int = 100) -> List[str]:
        """Generate domain suggestions using the trained model"""
        
        if not self.is_trained:
            logger.warning("Model not trained yet, using base model")
        
        # Format prompt
        prompt = f"Generate 3 domain name suggestions for this business:\n\nBusiness: {business_description}\n\nDomain suggestions:\n"
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config['data']['max_input_length']
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=self.config['generation']['temperature'],
                top_p=self.config['generation']['top_p'],
                do_sample=self.config['generation']['do_sample'],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract domain suggestions
        suggestions = self._extract_domains_from_text(generated_text, num_suggestions)
        
        return suggestions
    
    def _extract_domains_from_text(self, text: str, num_suggestions: int) -> List[str]:
        """Extract domain suggestions from generated text"""
        # Simple extraction - look for domains after the prompt
        lines = text.split('\n')
        domains = []
        
        for line in lines:
            line = line.strip()
            # Look for numbered domains or direct domain patterns
            if any(pattern in line.lower() for pattern in ['1.', '2.', '3.', '.com', '.net', '.org']):
                # Extract domain from line
                words = line.split()
                for word in words:
                    if '.' in word and any(tld in word.lower() for tld in ['.com', '.net', '.org', '.co']):
                        # Clean up the domain
                        domain = word.strip('.,!?()[]{}')
                        if len(domain) > 3:
                            domains.append(domain)
                            break
        
        # If no domains found, create simple fallback
        if not domains:
            business_words = business_description.lower().split()[:2]
            for i in range(num_suggestions):
                if i < len(business_words):
                    domains.append(f"{business_words[i]}.com")
                else:
                    domains.append(f"domain{i+1}.com")
        
        return domains[:num_suggestions]
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.model and self.tokenizer:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save config
            with open(save_path / 'model_config.json', 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"✅ Model saved to {save_path}")
        else:
            logger.warning("No model to save")
    
    def load_model(self, load_path: str):
        """Load a trained model"""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Model path not found: {load_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(load_path))
        
        # Load model
        self.model = PeftModel.from_pretrained(self.base_model, str(load_path))
        self.is_trained = True
        self.model_path = load_path
        
        logger.info(f"✅ Model loaded from {load_path}")
