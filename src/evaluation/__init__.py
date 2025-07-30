"""
LLM Evaluation Module

This module contains the LLMEvaluator class for evaluating domain name suggestion models
using LLM-as-a-Judge methodology with comprehensive metrics and analysis.
"""

import openai
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
import asyncio
import aiohttp
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
        
        logger.info(f"Initialized LLMJudgeEvaluator with {self.judge_model['provider']} as judge")
        
    def _setup_api_clients(self):
        """Setup API clients for different providers."""
        if self.judge_model['provider'] == 'openai':
            self.openai_client = openai.OpenAI()
        elif self.judge_model['provider'] == 'anthropic':
            self.anthropic_client = anthropic.Anthropic()
        # Add other providers as needed
            
    def evaluate_safety(self, business_description: str) -> Dict[str, Any]:
        """Evaluate if a business description is safe for domain generation."""
        prompt = self.prompts['safety_prompt'].format(
            business_description=business_description
        )
        
        try:
            if self.judge_model['provider'] == 'openai':
                response = self.openai_client.chat.completions.create(
                    model=self.judge_model['model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.judge_model['temperature'],
                    max_tokens=self.judge_model['max_tokens']
                )
                result_text = response.choices[0].message.content
                
            elif self.judge_model['provider'] == 'anthropic':
                response = self.anthropic_client.messages.create(
                    model=self.judge_model['model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.judge_model['temperature'],
                    max_tokens=self.judge_model['max_tokens']
                )
                result_text = response.content[0].text
            
            # Parse JSON response
            safety_result = json.loads(result_text)
            return safety_result
            
        except Exception as e:
            logger.error(f"Safety evaluation error: {e}")
            return {"is_safe": True, "reason": "evaluation_error"}
    
    def evaluate_domain_quality(self, 
                              business_description: str, 
                              domain_suggestions: List[str]) -> Dict[str, Any]:
        """Evaluate the quality of domain name suggestions."""
        
        # Format domain suggestions for prompt
        suggestions_text = "\n".join([f"{i+1}. {domain}" for i, domain in enumerate(domain_suggestions)])
        
        prompt = self.prompts['evaluation_prompt'].format(
            business_description=business_description,
            domain_suggestions=suggestions_text
        )
        
        try:
            if self.judge_model['provider'] == 'openai':
                response = self.openai_client.chat.completions.create(
                    model=self.judge_model['model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.judge_model['temperature'],
                    max_tokens=self.judge_model['max_tokens']
                )
                result_text = response.choices[0].message.content
                
            elif self.judge_model['provider'] == 'anthropic':
                response = self.anthropic_client.messages.create(
                    model=self.judge_model['model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.judge_model['temperature'],
                    max_tokens=self.judge_model['max_tokens']
                )
                result_text = response.content[0].text
            
            # Parse the structured response
            evaluation_result = self._parse_evaluation_response(result_text)
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Domain quality evaluation error: {e}")
            return self._default_evaluation_result(domain_suggestions)
    
    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM judge response into structured format."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'{.*}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                # Fallback parsing logic
                return self._fallback_parse_response(response_text)
                
        except json.JSONDecodeError:
            return self._fallback_parse_response(response_text)
    
    def _fallback_parse_response(self, response_text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails."""
        # Extract numerical scores using regex
        scores = {}
        criteria = ['relevance', 'memorability', 'brandability', 'technical_validity', 
                   'length_appropriateness', 'uniqueness']
        
        for criterion in criteria:
            pattern = rf'{criterion}[:\s]*(\d+(?:\.\d+)?)'
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                scores[criterion] = float(match.group(1))
            else:
                scores[criterion] = 3.0  # Default score
        
        return {
            "individual_scores": scores,
            "overall_score": sum(scores.values()) / len(scores),
            "explanation": "Parsed from unstructured response"
        }
    
    def _default_evaluation_result(self, domain_suggestions: List[str]) -> Dict[str, Any]:
        """Return default evaluation result in case of errors."""
        default_scores = {criterion['name']: 3.0 for criterion in self.metrics}
        return {
            "individual_scores": default_scores,
            "overall_score": 3.0,
            "explanation": "Default scores due to evaluation error"
        }
    
    def evaluate_batch(self, 
                      evaluation_data: List[Dict[str, Any]], 
                      include_safety: bool = True) -> List[EvaluationResult]:
        """Evaluate a batch of domain suggestions."""
        
        results = []
        
        for i, data in enumerate(evaluation_data):
            business_description = data['business_description']
            domain_suggestions = data['domain_suggestions']
            
            logger.info(f"Evaluating {i+1}/{len(evaluation_data)}: {business_description[:50]}...")
            
            # Safety evaluation
            is_safe = True
            safety_reason = None
            if include_safety:
                safety_result = self.evaluate_safety(business_description)
                is_safe = safety_result.get('is_safe', True)
                safety_reason = safety_result.get('reason', None)
            
            # Quality evaluation (only if safe)
            if is_safe:
                quality_result = self.evaluate_domain_quality(business_description, domain_suggestions)
                scores = quality_result.get('individual_scores', {})
                overall_score = quality_result.get('overall_score', 0.0)
                explanation = quality_result.get('explanation', '')
            else:
                scores = {criterion['name']: 0.0 for criterion in self.metrics}
                overall_score = 0.0
                explanation = f"Blocked due to safety concerns: {safety_reason}"
            
            # Create evaluation result
            result = EvaluationResult(
                domain_suggestions=domain_suggestions,
                scores=scores,
                overall_score=overall_score,
                explanation=explanation,
                is_safe=is_safe,
                safety_reason=safety_reason
            )
            
            results.append(result)
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        return results
    
    def calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate aggregate metrics from evaluation results."""
        
        if not results:
            return {}
        
        # Filter safe results for quality metrics
        safe_results = [r for r in results if r.is_safe]
        
        metrics = {}
        
        # Safety metrics
        metrics['safety_metrics'] = {
            'total_evaluated': len(results),
            'safe_count': len(safe_results),
            'unsafe_count': len(results) - len(safe_results),
            'safety_rate': len(safe_results) / len(results) if results else 0
        }
        
        if safe_results:
            # Quality metrics
            all_scores = {}
            for criterion in self.metrics:
                criterion_name = criterion['name']
                scores = [r.scores.get(criterion_name, 0) for r in safe_results]
                all_scores[criterion_name] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'std': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'min': min(scores),
                    'max': max(scores)
                }
            
            # Overall scores
            overall_scores = [r.overall_score for r in safe_results]
            metrics['quality_metrics'] = {
                'criterion_scores': all_scores,
                'overall_score': {
                    'mean': statistics.mean(overall_scores),
                    'median': statistics.median(overall_scores),
                    'std': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                    'min': min(overall_scores),  
                    'max': max(overall_scores)
                }
            }
            
            # Performance categorization
            excellent_count = sum(1 for score in overall_scores if score >= self.config['thresholds']['excellent_score'])
            acceptable_count = sum(1 for score in overall_scores if score >= self.config['thresholds']['minimum_acceptable_score'])
            
            metrics['performance_distribution'] = {
                'excellent_rate': excellent_count / len(safe_results),
                'acceptable_rate': acceptable_count / len(safe_results),
                'below_threshold_rate': 1 - (acceptable_count / len(safe_results))
            }
        
        return metrics
    
    def save_evaluation_results(self, 
                              results: List[EvaluationResult], 
                              output_path: str,
                              metadata: Dict[str, Any] = None):
        """Save evaluation results to file."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'domain_suggestions': result.domain_suggestions,
                'scores': result.scores,
                'overall_score': result.overall_score,
                'explanation': result.explanation,
                'is_safe': result.is_safe,
                'safety_reason': result.safety_reason
            })
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        # Prepare final output
        output_data = {
            'metadata': metadata or {},
            'evaluation_config': self.config,
            'results': serializable_results,
            'aggregate_metrics': aggregate_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save as JSON
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Save summary as CSV
        df_data = []
        for i, result in enumerate(results):
            row = {
                'id': i,
                'overall_score': result.overall_score,
                'is_safe': result.is_safe,
                'num_suggestions': len(result.domain_suggestions),
                **result.scores
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        logger.info(f"Evaluation results saved to {output_path}")
        
        return aggregate_metrics
    
    def compare_models(self, 
                      baseline_results: List[EvaluationResult],
                      improved_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compare evaluation results between two models."""
        
        baseline_metrics = self.calculate_aggregate_metrics(baseline_results)
        improved_metrics = self.calculate_aggregate_metrics(improved_results)
        
        comparison = {
            'baseline_metrics': baseline_metrics,
            'improved_metrics': improved_metrics,
            'improvements': {}
        }
        
        # Calculate improvements
        if baseline_metrics and improved_metrics:
            baseline_overall = baseline_metrics['quality_metrics']['overall_score']['mean']
            improved_overall = improved_metrics['quality_metrics']['overall_score']['mean']
            
            improvement = improved_overall - baseline_overall
            improvement_pct = (improvement / baseline_overall) * 100 if baseline_overall > 0 else 0
            
            comparison['improvements'] = {
                'overall_score_improvement': improvement,
                'overall_score_improvement_pct': improvement_pct,
                'statistical_significance': self._test_statistical_significance(
                    [r.overall_score for r in baseline_results if r.is_safe],
                    [r.overall_score for r in improved_results if r.is_safe]
                )
            }
        
        return comparison
    
    def _test_statistical_significance(self, baseline_scores: List[float], improved_scores: List[float]) -> Dict[str, Any]:
        """Test statistical significance of improvement."""
        from scipy import stats
        
        if len(baseline_scores) < 2 or len(improved_scores) < 2:
            return {"test": "insufficient_data", "significant": False}
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(improved_scores, baseline_scores)
        
        return {
            "test": "independent_t_test",
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "effect_size": abs(statistics.mean(improved_scores) - statistics.mean(baseline_scores)) / 
                          statistics.stdev(baseline_scores + improved_scores)
        }

# Example usage
if __name__ == "__main__":
    evaluator = LLMJudgeEvaluator()
    
    # Example evaluation data
    test_data = [
        {
            "business_description": "organic coffee shop in downtown",
            "domain_suggestions": ["organicbeans.com", "downtowncoffee.net", "freshbrew.org"]
        },
        {
            "business_description": "tech startup focused on AI solutions",
            "domain_suggestions": ["aitech.com", "smartsolutions.ai", "techstartup.co"]
        }
    ]
    
    # Note: This requires valid API keys to run
    try:
        results = evaluator.evaluate_batch(test_data)
        metrics = evaluator.calculate_aggregate_metrics(results)
        print("Evaluation completed successfully!")
        print(f"Average overall score: {metrics.get('quality_metrics', {}).get('overall_score', {}).get('mean', 'N/A')}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("Please ensure API keys are properly configured.")
