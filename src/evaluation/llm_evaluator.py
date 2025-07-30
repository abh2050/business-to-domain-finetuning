"""
LLM Evaluation Module

This module contains the LLMEvaluator class for evaluating domain name suggestion models
using LLM-as-a-Judge methodology with comprehensive metrics and analysis.
"""

import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import re
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """
    LLM-as-a-Judge evaluator for domain name suggestion models.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the LLMEvaluator.
        
        Args:
            config_path: Path to the evaluation configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.client = None
        
        # Initialize OpenAI client if API key is provided
        self._setup_openai_client()
        
        # Evaluation metrics storage
        self.evaluation_results = {}
        self.detailed_scores = []
        
        logger.info(f"Initialized LLMEvaluator with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_openai_client(self):
        """Setup OpenAI client for LLM-as-a-Judge evaluation."""
        api_key = self.config.get('judge_model', {}).get('api_key')
        
        if api_key and api_key != "your_openai_api_key_here":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            except ImportError:
                logger.warning("OpenAI package not available. Using mock evaluation.")
                self.client = None
        else:
            logger.info("No valid OpenAI API key provided. Using mock evaluation.")
            self.client = None
    
    def evaluate_model_predictions(self, predictions: List[Dict[str, Any]], model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate model predictions using multiple metrics.
        
        Args:
            predictions: List of predictions with business_description, ground_truth, and predictions fields
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Starting evaluation for {model_name} with {len(predictions)} predictions")
        
        # Initialize metrics
        metrics = {
            'overall_score': 0.0,
            'relevance': 0.0,
            'creativity': 0.0,
            'professionalism': 0.0,
            'technical': 0.0,
            'total_predictions': len(predictions),
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        individual_scores = []
        
        # Evaluate each prediction
        for i, pred in enumerate(predictions):
            if (i + 1) % 10 == 0:
                logger.info(f"Evaluating prediction {i+1}/{len(predictions)}")
            
            # Get individual scores for this prediction
            scores = self._evaluate_single_prediction(pred)
            individual_scores.append(scores)
            
            # Add to running totals
            for metric in ['relevance', 'creativity', 'professionalism', 'technical']:
                metrics[metric] += scores.get(metric, 0.0)
        
        # Calculate averages
        for metric in ['relevance', 'creativity', 'professionalism', 'technical']:
            metrics[metric] /= len(predictions)
        
        # Calculate overall score (weighted average)
        weights = {
            'relevance': 0.3,
            'creativity': 0.25,
            'professionalism': 0.25,
            'technical': 0.20
        }
        
        metrics['overall_score'] = (
            metrics['relevance'] * weights['relevance'] +
            metrics['creativity'] * weights['creativity'] +
            metrics['professionalism'] * weights['professionalism'] +
            metrics['technical'] * weights['technical']
        )
        
        # Store detailed results
        self.evaluation_results[model_name] = metrics
        self.detailed_scores.extend(individual_scores)
        
        logger.info(f"Evaluation completed. Overall score: {metrics['overall_score']:.3f}")
        return metrics
    
    def _evaluate_single_prediction(self, prediction: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single prediction using LLM-as-a-Judge or heuristic methods.
        
        Args:
            prediction: Dictionary with business_description, ground_truth, and predictions fields
            
        Returns:
            Dictionary with individual scores
        """
        business_desc = prediction.get('business_description', '')
        ground_truth = prediction.get('ground_truth', [])
        predicted = prediction.get('predictions', [])
        
        if self.client:
            # Use OpenAI for evaluation
            return self._llm_judge_evaluation(business_desc, ground_truth, predicted)
        else:
            # Use heuristic evaluation
            return self._heuristic_evaluation(business_desc, ground_truth, predicted)
    
    def _llm_judge_evaluation(self, business_desc: str, ground_truth: List[str], predicted: List[str]) -> Dict[str, float]:
        """
        Use LLM-as-a-Judge for evaluation.
        
        Args:
            business_desc: Business description
            ground_truth: Ground truth domain suggestions  
            predicted: Predicted domain suggestions
            
        Returns:
            Dictionary with scores for different aspects
        """
        try:
            prompt = self._create_judge_prompt(business_desc, ground_truth, predicted)
            
            response = self.client.chat.completions.create(
                model=self.config['judge_model']['model_name'],
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of domain name suggestions. Provide objective scores based on the criteria given."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['judge_model']['temperature'],
                max_tokens=self.config['judge_model']['max_tokens']
            )
            
            # Parse the response to extract scores
            response_text = response.choices[0].message.content
            scores = self._parse_judge_response(response_text)
            
            return scores
            
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}. Falling back to heuristic evaluation.")
            return self._heuristic_evaluation(business_desc, ground_truth, predicted)
    
    def _create_judge_prompt(self, business_desc: str, ground_truth: List[str], predicted: List[str]) -> str:
        """Create the prompt for LLM-as-a-Judge evaluation."""
        prompt = f"""
Please evaluate the quality of domain name suggestions for the following business:

Business Description: {business_desc}

Reference Domain Suggestions: {', '.join(ground_truth[:5])}
Generated Domain Suggestions: {', '.join(predicted[:5])}

Please rate the generated suggestions on a scale of 1-10 for each of the following criteria:

1. RELEVANCE (1-10): How well do the suggested domains relate to the business description?
2. CREATIVITY (1-10): How creative and unique are the domain suggestions?
3. PROFESSIONALISM (1-10): How professional and business-appropriate are these domains?
4. TECHNICAL (1-10): Are the domains technically sound (proper format, length, characters)?

Provide your response in this exact format:
RELEVANCE: X/10
CREATIVITY: X/10  
PROFESSIONALISM: X/10
TECHNICAL: X/10

Brief explanation of your scoring:
"""
        return prompt
    
    def _parse_judge_response(self, response_text: str) -> Dict[str, float]:
        """
        Parse the LLM judge response to extract numerical scores.
        
        Args:
            response_text: The response from the LLM judge
            
        Returns:
            Dictionary with parsed scores
        """
        scores = {}
        
        # Look for score patterns
        patterns = {
            'relevance': r'RELEVANCE:\s*(\d+(?:\.\d+)?)',
            'creativity': r'CREATIVITY:\s*(\d+(?:\.\d+)?)',
            'professionalism': r'PROFESSIONALISM:\s*(\d+(?:\.\d+)?)',
            'technical': r'TECHNICAL:\s*(\d+(?:\.\d+)?)'
        }
        
        for metric, pattern in patterns.items():
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                # Normalize to 0-1 scale
                scores[metric] = min(max(score / 10.0, 0.0), 1.0)
            else:
                # Default score if not found
                scores[metric] = 0.5
        
        return scores
    
    def _heuristic_evaluation(self, business_desc: str, ground_truth: List[str], predicted: List[str]) -> Dict[str, float]:
        """
        Heuristic evaluation when LLM-as-a-Judge is not available.
        
        Args:
            business_desc: Business description
            ground_truth: Ground truth domain suggestions
            predicted: Predicted domain suggestions
            
        Returns:
            Dictionary with heuristic scores
        """
        scores = {}
        
        # 1. Relevance: Check keyword overlap
        business_words = set(re.findall(r'\b\w+\b', business_desc.lower()))
        predicted_words = set()
        for domain in predicted:
            predicted_words.update(re.findall(r'\b\w+\b', domain.lower().replace('.', ' ')))
        
        relevance = len(business_words.intersection(predicted_words)) / max(len(business_words), 1)
        scores['relevance'] = min(relevance * 2, 1.0)  # Scale and cap at 1.0
        
        # 2. Technical Quality: Check domain format and length
        technical_score = 0
        valid_domains = 0
        
        for domain in predicted:
            domain_score = 0
            
            # Check if it looks like a domain
            if '.' in domain and len(domain.split('.')) >= 2:
                domain_score += 0.3
                
                # Check length (ideal 5-15 characters before TLD)
                name_part = domain.split('.')[0]
                if 5 <= len(name_part) <= 15:
                    domain_score += 0.3
                elif len(name_part) <= 20:
                    domain_score += 0.2
                
                # Check for hyphens (slightly penalize)
                if '-' not in domain:
                    domain_score += 0.2
                
                # Check for numbers (slightly penalize unless it makes sense)
                if not any(char.isdigit() for char in domain):
                    domain_score += 0.2
            
            technical_score += domain_score
            if domain_score > 0:
                valid_domains += 1
        
        scores['technical'] = technical_score / max(len(predicted), 1)
        
        # 3. Creativity: Penalize very common patterns
        creativity_score = 0.7  # Base creativity score
        common_patterns = ['company', 'business', 'corp', 'inc', 'llc']
        
        for domain in predicted:
            has_common = any(pattern in domain.lower() for pattern in common_patterns)
            if not has_common:
                creativity_score += 0.1
        
        scores['creativity'] = min(creativity_score, 1.0)
        
        # 4. Professionalism: Shorter, clean domains score higher
        professionalism_score = 0
        for domain in predicted:
            name_part = domain.split('.')[0] if '.' in domain else domain
            domain_prof = 0
            
            # Length factor
            if len(name_part) <= 8:
                domain_prof += 0.4
            elif len(name_part) <= 12:
                domain_prof += 0.3
            elif len(name_part) <= 16:
                domain_prof += 0.2
            
            # Avoid numbers and hyphens for professionalism  
            if not any(char.isdigit() for char in name_part):
                domain_prof += 0.3
            if '-' not in name_part:
                domain_prof += 0.3
            
            professionalism_score += domain_prof
        
        scores['professionalism'] = professionalism_score / max(len(predicted), 1)
        
        return scores
    
    def analyze_by_business_type(self, predictions: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze performance by business type.
        
        Args:
            predictions: List of predictions
            metrics: Overall metrics
            
        Returns:
            Dictionary mapping business type to performance score
        """
        business_type_scores = defaultdict(list)
        
        for pred in predictions:
            business_type = pred.get('business_type', 'unknown')
            
            # Get the individual score for this prediction
            individual_scores = self._evaluate_single_prediction(pred)
            overall_score = sum(individual_scores.values()) / len(individual_scores)
            
            business_type_scores[business_type].append(overall_score)
        
        # Calculate average scores per business type
        avg_scores = {}
        for business_type, scores in business_type_scores.items():
            avg_scores[business_type] = np.mean(scores)
        
        # Sort by score (descending)
        sorted_scores = dict(sorted(avg_scores.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_scores
    
    def analyze_edge_cases(self, predictions: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze performance on edge cases vs regular cases.
        
        Args:
            predictions: List of predictions
            metrics: Overall metrics
            
        Returns:
            Dictionary with edge case analysis
        """
        edge_case_scores = []
        regular_case_scores = []
        
        for pred in predictions:
            # Check if this is an edge case
            is_edge_case = pred.get('is_edge_case', False)
            
            # Get individual score
            individual_scores = self._evaluate_single_prediction(pred)
            overall_score = sum(individual_scores.values()) / len(individual_scores)
            
            if is_edge_case:
                edge_case_scores.append(overall_score)
            else:
                regular_case_scores.append(overall_score)
        
        # Calculate averages
        regular_avg = np.mean(regular_case_scores) if regular_case_scores else 0.0
        edge_case_avg = np.mean(edge_case_scores) if edge_case_scores else 0.0
        
        return {
            'regular': regular_avg,
            'edge_cases': edge_case_avg,
            'gap': regular_avg - edge_case_avg,
            'num_regular': len(regular_case_scores),
            'num_edge_cases': len(edge_case_scores)
        }
    
    def generate_evaluation_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Report as a string
        """
        if not self.evaluation_results:
            return "No evaluation results available. Run evaluate_model_predictions() first."
        
        report_lines = []
        report_lines.append("# Domain Name Model Evaluation Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Overall results
        for model_name, metrics in self.evaluation_results.items():
            report_lines.append(f"## Model: {model_name}")
            report_lines.append("")
            report_lines.append(f"**Overall Score:** {metrics['overall_score']:.3f}")
            report_lines.append(f"**Total Predictions:** {metrics['total_predictions']}")
            report_lines.append("")
            
            report_lines.append("### Detailed Metrics")
            report_lines.append(f"- **Relevance:** {metrics['relevance']:.3f}")
            report_lines.append(f"- **Creativity:** {metrics['creativity']:.3f}")
            report_lines.append(f"- **Professionalism:** {metrics['professionalism']:.3f}")
            report_lines.append(f"- **Technical:** {metrics['technical']:.3f}")
            report_lines.append("")
        
        # Additional analyses would go here
        report_lines.append("### Notes")
        report_lines.append("- Scores are normalized to 0-1 range")
        report_lines.append("- Overall score is a weighted average of individual metrics")
        if not self.client:
            report_lines.append("- Evaluation performed using heuristic methods (LLM judge not available)")
        else:
            report_lines.append("- Evaluation performed using LLM-as-a-Judge methodology")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to: {output_path}")
        
        return report_text
