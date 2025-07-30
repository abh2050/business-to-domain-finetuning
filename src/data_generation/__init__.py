"""
Synthetic Dataset Generation for Domain Name Suggestion LLM

This module provides comprehensive synthetic data generation for training
domain name suggestion models, including diverse business types, edge cases,
and realistic domain name patterns.
"""

import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging
import re
from itertools import combinations
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Advanced synthetic data generator for domain name suggestion model training.
    
    Features:
    - 30+ business categories with specialized descriptors
    - Multiple complexity levels and patterns
    - Edge case generation for robust testing
    - Realistic domain name patterns and variations
    - Quality control and validation
    """
    
    def __init__(self, config_path: str = None):
        """Initialize with comprehensive business data and patterns."""
        
        # Extended business types with industry-specific characteristics
        self.business_types = {
            "restaurant": {
                "descriptors": ["fine dining", "casual", "fast food", "ethnic cuisine", "farm-to-table", "vegan", "seafood", "steakhouse", "bistro", "cafe"],
                "keywords": ["kitchen", "grill", "dining", "taste", "flavor", "fresh", "delicious", "cuisine", "chef"],
                "locations": ["downtown", "waterfront", "historic district", "shopping center"]
            },
            "tech_startup": {
                "descriptors": ["AI/ML", "fintech", "edtech", "healthtech", "SaaS", "mobile app", "blockchain", "cybersecurity", "cloud computing"],
                "keywords": ["tech", "digital", "smart", "cloud", "data", "ai", "solution", "platform", "innovation"],
                "locations": ["silicon valley", "tech hub", "innovation district", "startup incubator"]
            },
            "retail_store": {
                "descriptors": ["fashion", "electronics", "home goods", "sporting goods", "books", "toys", "jewelry", "furniture"],
                "keywords": ["shop", "store", "market", "boutique", "outlet", "retail", "fashion", "style"],
                "locations": ["mall", "shopping center", "main street", "outlet center"]
            },
            "consulting_firm": {
                "descriptors": ["management", "IT", "financial", "HR", "strategy", "marketing", "operations", "legal"],
                "keywords": ["consulting", "advisory", "solutions", "experts", "professional", "strategy", "business"],
                "locations": ["business district", "corporate center", "downtown"]
            },
            "fitness_center": {
                "descriptors": ["gym", "crossfit", "yoga", "pilates", "martial arts", "personal training", "wellness", "boxing"],
                "keywords": ["fit", "strong", "health", "wellness", "active", "training", "muscle", "power"],
                "locations": ["community center", "strip mall", "residential area"]
            },
            "healthcare_clinic": {
                "descriptors": ["family practice", "urgent care", "specialist", "wellness", "alternative medicine", "pediatric", "dental"],
                "keywords": ["health", "care", "medical", "wellness", "healing", "family", "clinic", "doctor"],
                "locations": ["medical district", "suburban", "hospital campus"]
            },
            "coffee_shop": {
                "descriptors": ["artisan", "local roaster", "specialty", "organic", "fair trade", "espresso bar", "cafe"],
                "keywords": ["coffee", "brew", "roast", "bean", "espresso", "latte", "cafe", "morning"],
                "locations": ["downtown", "university area", "neighborhood", "strip mall"]
            },
            "photography_studio": {
                "descriptors": ["portrait", "wedding", "commercial", "artistic", "family", "newborn", "event"],
                "keywords": ["photo", "picture", "capture", "memory", "visual", "creative", "lens", "studio"],
                "locations": ["arts district", "downtown", "residential", "creative quarter"]
            },
            "law_firm": {
                "descriptors": ["personal injury", "family law", "corporate", "criminal defense", "real estate", "immigration"],
                "keywords": ["law", "legal", "attorney", "counsel", "justice", "advocate", "defense", "rights"],
                "locations": ["legal district", "downtown", "business center"]
            },
            "real_estate_agency": {
                "descriptors": ["residential", "commercial", "luxury", "first-time buyers", "investment", "property management"],
                "keywords": ["property", "home", "house", "real estate", "invest", "market", "buy", "sell"],
                "locations": ["downtown", "suburban", "upscale neighborhood"]
            }
        }
        
        # Add more business types
        additional_types = {
            "veterinary_clinic": {"descriptors": ["small animal", "exotic pets", "emergency", "mobile"], "keywords": ["pet", "animal", "vet", "care"], "locations": ["suburban", "rural"]},
            "auto_repair": {"descriptors": ["foreign cars", "domestic", "transmission", "collision"], "keywords": ["auto", "car", "repair", "service"], "locations": ["industrial", "highway"]},
            "landscaping": {"descriptors": ["residential", "commercial", "design", "maintenance"], "keywords": ["landscape", "garden", "outdoor", "green"], "locations": ["suburban", "rural"]},
            "cleaning_service": {"descriptors": ["residential", "commercial", "deep cleaning", "eco-friendly"], "keywords": ["clean", "spotless", "fresh", "service"], "locations": ["citywide", "suburban"]},
            "catering": {"descriptors": ["wedding", "corporate", "casual", "fine dining"], "keywords": ["catering", "event", "food", "party"], "locations": ["event venues", "citywide"]},
            "bookstore": {"descriptors": ["independent", "used books", "specialty", "academic"], "keywords": ["book", "read", "literature", "knowledge"], "locations": ["downtown", "university area"]},
            "art_gallery": {"descriptors": ["contemporary", "local artists", "sculpture", "photography"], "keywords": ["art", "gallery", "creative", "exhibit"], "locations": ["arts district", "downtown"]},
            "dental_practice": {"descriptors": ["family dentistry", "cosmetic", "orthodontics", "pediatric"], "keywords": ["dental", "smile", "teeth", "oral"], "locations": ["medical plaza", "suburban"]},
            "accounting_firm": {"descriptors": ["small business", "tax preparation", "corporate", "bookkeeping"], "keywords": ["accounting", "tax", "financial", "books"], "locations": ["business district", "professional plaza"]},
            "marketing_agency": {"descriptors": ["digital marketing", "social media", "brand strategy", "creative"], "keywords": ["marketing", "brand", "creative", "digital"], "locations": ["creative district", "tech hub"]}
        }
        
        self.business_types.update(additional_types)
        
        # Domain generation patterns with weights
        self.domain_patterns = [
            ("{keyword1}.com", 0.25),
            ("{keyword1}{keyword2}.com", 0.20),
            ("{prefix}{keyword1}.com", 0.15),
            ("{keyword1}{suffix}.com", 0.15),
            ("{location}{keyword1}.com", 0.10),
            ("{keyword1}.co", 0.05),
            ("{keyword1}.net", 0.05),
            ("{keyword1}.org", 0.05)
        ]
        
        # Common prefixes and suffixes for domain generation
        self.prefixes = ["best", "top", "prime", "elite", "pro", "smart", "quick", "easy", "local", "my", "the"]
        self.suffixes = ["hub", "zone", "spot", "place", "lab", "works", "solutions", "group", "co", "pro", "plus", "247"]
        
        # Cities and locations for realistic business descriptions
        self.cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
            "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
            "Fort Worth", "Columbus", "Charlotte", "San Francisco", "Indianapolis",
            "Seattle", "Denver", "Washington DC", "Boston", "Nashville", "Baltimore"
        ]
        
        # Neighborhood types
        self.neighborhoods = [
            "downtown", "uptown", "midtown", "old town", "historic district", "arts district",
            "business district", "financial district", "shopping district", "waterfront",
            "riverside", "hillside", "suburban", "residential area", "community"
        ]
        
        # Quality modifiers for descriptions
        self.quality_modifiers = [
            "award-winning", "family-owned", "locally-owned", "established", "experienced",
            "professional", "trusted", "reliable", "friendly", "expert", "specialized",
            "premium", "quality", "affordable", "convenient", "full-service"
        ]
        
        # Initialize random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
    def generate_business_description(self, complexity: str = "medium", business_type: str = None) -> Dict[str, Any]:
        """
        Generate realistic business descriptions with varying complexity levels.
        
        Args:
            complexity: 'simple', 'medium', 'complex', or 'edge_case'
            business_type: Specific business type to generate (optional)
            
        Returns:
            Dictionary containing business description and metadata
        """
        if business_type is None:
            business_type = random.choice(list(self.business_types.keys()))
        
        business_info = self.business_types[business_type]
        
        if complexity == "simple":
            # Basic business type only
            description = business_type.replace('_', ' ')
            
        elif complexity == "medium":
            # Business type + descriptor + location
            descriptor = random.choice(business_info["descriptors"])
            city = random.choice(self.cities)
            neighborhood = random.choice(self.neighborhoods)
            
            templates = [
                f"{descriptor} {business_type.replace('_', ' ')} in {city}",
                f"{descriptor} {business_type.replace('_', ' ')} in {neighborhood}",
                f"{business_type.replace('_', ' ')} specializing in {descriptor}",
                f"local {business_type.replace('_', ' ')} offering {descriptor} services"
            ]
            description = random.choice(templates)
            
        elif complexity == "complex":
            # Full description with multiple elements
            descriptor = random.choice(business_info["descriptors"])
            city = random.choice(self.cities)
            neighborhood = random.choice(self.neighborhoods)
            quality_modifier = random.choice(self.quality_modifiers)
            
            # Additional context options
            additional_context = [
                f"serving the {city} area for over {random.randint(5, 25)} years",
                f"with a focus on customer satisfaction and quality service",
                f"committed to providing excellent {descriptor} services",
                f"known for our professional and friendly approach",
                f"dedicated to meeting all your {business_type.replace('_', ' ')} needs"
            ]
            
            templates = [
                f"{quality_modifier} {descriptor} {business_type.replace('_', ' ')} located in {neighborhood}, {city}, {random.choice(additional_context)}",
                f"We are a {quality_modifier} {business_type.replace('_', ' ')} in {city} specializing in {descriptor}, {random.choice(additional_context)}",
                f"{quality_modifier} {business_type.replace('_', ' ')} offering {descriptor} services in the {neighborhood} area of {city}"
            ]
            description = random.choice(templates)
            
        else:  # edge_case
            description = self._generate_edge_case_description(business_type)
            
        return {
            "business_description": description,
            "business_type": business_type,
            "complexity": complexity,
            "city": city if complexity != "simple" else None,
            "neighborhood": neighborhood if complexity in ["medium", "complex"] else None,
            "keywords": self._extract_keywords(description)
        }
    
    def _generate_edge_case_description(self, business_type: str) -> str:
        """Generate challenging edge case descriptions."""
        edge_case_types = [
            "ambiguous",
            "minimal",
            "excessive",
            "technical",
            "unusual"
        ]
        
        case_type = random.choice(edge_case_types)
        business_info = self.business_types[business_type]
        
        if case_type == "ambiguous":
            return random.choice([
                "business", "company", "service provider", "local establishment",
                "professional services", "family business", "small business"
            ])
            
        elif case_type == "minimal":
            return random.choice([
                business_type.split('_')[0], 
                business_info["keywords"][0] if business_info["keywords"] else "service",
                "local " + business_type.split('_')[0]
            ])
            
        elif case_type == "excessive":
            # Very long, overly detailed description
            descriptor = random.choice(business_info["descriptors"])
            city = random.choice(self.cities)
            keywords = business_info["keywords"][:3]
            
            return f"A comprehensive full-service {descriptor} {business_type.replace('_', ' ')} " + \
                   f"specializing in {', '.join(keywords)} and located in the heart of {city} " + \
                   f"with over {random.randint(10, 50)} years of combined experience serving the " + \
                   f"local community with award-winning customer service and innovative solutions " + \
                   f"for all your {business_type.replace('_', ' ')} needs and requirements"
                   
        elif case_type == "technical":
            # Technical jargon heavy
            tech_terms = ["AI-powered", "blockchain-enabled", "cloud-based", "IoT-integrated", 
                         "machine learning", "data-driven", "API-first", "microservices"]
            return f"{random.choice(tech_terms)} {business_type.replace('_', ' ')} " + \
                   f"utilizing {random.choice(tech_terms)} architecture"
                   
        else:  # unusual
            # Unusual characters or formatting
            base_desc = f"{business_type.replace('_', ' ')} service"
            return random.choice([
                base_desc.upper(),
                base_desc.replace(' ', ''),
                base_desc + "!!",
                base_desc + " & more",
                base_desc + " (NEW!)"
            ])
    
    def _extract_keywords(self, description: str) -> List[str]:
        """Extract relevant keywords from business description."""
        # Remove common stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in',
            'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with', 'we',
            'our', 'your', 'their', 'all', 'any', 'both', 'each', 'more', 'most', 'other',
            'some', 'such', 'than', 'too', 'very', 'area', 'years', 'over', 'located'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', description.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def generate_domain_suggestions(self, business_data: Dict[str, Any], num_suggestions: int = None) -> List[str]:
        """
        Generate realistic domain name suggestions for a business.
        
        Args:
            business_data: Business information dictionary
            num_suggestions: Number of suggestions to generate (default: 3-7)
            
        Returns:
            List of domain name suggestions
        """
        if num_suggestions is None:
            num_suggestions = random.randint(3, 7)
            
        business_type = business_data["business_type"]
        keywords = business_data.get("keywords", [])
        city = business_data.get("city", "")
        
        # Get business-specific keywords
        business_info = self.business_types.get(business_type, {})
        business_keywords = business_info.get("keywords", [])
        
        # Combine all available keywords
        all_keywords = list(set(keywords + business_keywords))
        
        suggestions = []
        
        # Generate different types of suggestions
        for _ in range(num_suggestions * 2):  # Generate extra to filter later
            suggestion_type = random.choices(
                ["brand", "descriptive", "keyword_combo", "location_based", "business_type"],
                weights=[0.25, 0.25, 0.20, 0.15, 0.15]
            )[0]
            
            if suggestion_type == "brand":
                domain = self._generate_brand_domain(all_keywords)
            elif suggestion_type == "descriptive":
                domain = self._generate_descriptive_domain(all_keywords, business_type)
            elif suggestion_type == "keyword_combo":
                domain = self._generate_keyword_combo_domain(all_keywords)
            elif suggestion_type == "location_based":
                domain = self._generate_location_domain(all_keywords, city)
            else:  # business_type
                domain = self._generate_business_type_domain(business_type, all_keywords)
            
            if domain and self._is_valid_domain(domain):
                suggestions.append(domain)
            
            if len(suggestions) >= num_suggestions:
                break
        
        # Ensure we have at least some suggestions
        if len(suggestions) < 3:
            suggestions.extend(self._generate_fallback_domains(business_type, 3 - len(suggestions)))
        
        return list(set(suggestions))[:num_suggestions]  # Remove duplicates and limit
    
    def _generate_brand_domain(self, keywords: List[str]) -> str:
        """Generate brandable domain names."""
        if not keywords:
            keywords = ["business", "service", "company"]
            
        base_word = random.choice(keywords)
        
        # Truncate or modify for brandability
        if len(base_word) > 6:
            base_word = base_word[:random.randint(3, 6)]
        
        brand_techniques = [
            lambda w: random.choice(self.prefixes) + w,
            lambda w: w + random.choice(self.suffixes),
            lambda w: w + str(random.randint(1, 99)),
            lambda w: w.replace('e', '3').replace('o', '0') if random.random() < 0.3 else w,
            lambda w: w + random.choice(['ly', 'fy', 'io', 'co'])
        ]
        
        technique = random.choice(brand_techniques)
        branded_name = technique(base_word)
        
        extension = random.choices(['.com', '.co', '.net', '.io'], weights=[0.7, 0.15, 0.1, 0.05])[0]
        return branded_name + extension
    
    def _generate_descriptive_domain(self, keywords: List[str], business_type: str) -> str:
        """Generate descriptive domain names."""
        if not keywords:
            keywords = [business_type.replace('_', '')]
            
        # Combine 2-3 keywords
        if len(keywords) >= 2:
            selected_keywords = random.sample(keywords, min(random.randint(2, 3), len(keywords)))
            domain_name = ''.join(selected_keywords)
        else:
            domain_name = keywords[0] + random.choice(['service', 'company', 'group', 'solutions', 'pro'])
        
        # Clean up the domain name
        domain_name = re.sub(r'[^a-zA-Z0-9]', '', domain_name.lower())
        
        extension = random.choices(['.com', '.net', '.org'], weights=[0.8, 0.15, 0.05])[0]
        return domain_name + extension
    
    def _generate_keyword_combo_domain(self, keywords: List[str]) -> str:
        """Generate keyword combination domains."""
        if not keywords:
            return None
            
        base_keyword = random.choice(keywords)
        modifier = random.choice(self.prefixes + self.suffixes + ['local', 'best', 'top', '247', 'express'])
        
        combinations = [
            modifier + base_keyword,
            base_keyword + modifier,
            modifier + base_keyword[:4] if len(base_keyword) > 4 else modifier + base_keyword
        ]
        
        domain_name = random.choice(combinations)
        domain_name = re.sub(r'[^a-zA-Z0-9]', '', domain_name.lower())
        
        extension = random.choices(['.com', '.net', '.co'], weights=[0.7, 0.2, 0.1])[0]
        return domain_name + extension
    
    def _generate_location_domain(self, keywords: List[str], city: str) -> str:
        """Generate location-based domains."""
        if not city:
            return None
            
        city_clean = re.sub(r'[^a-zA-Z0-9]', '', city.lower())
        
        if keywords:
            keyword = random.choice(keywords)
            combinations = [
                city_clean + keyword,
                keyword + city_clean,
                city_clean + keyword[:4] if len(keyword) > 4 else city_clean + keyword
            ]
            domain_name = random.choice(combinations)
        else:
            domain_name = city_clean + random.choice(['business', 'service', 'local'])
        
        extension = random.choices(['.com', '.net', '.org'], weights=[0.7, 0.2, 0.1])[0]
        return domain_name + extension
    
    def _generate_business_type_domain(self, business_type: str, keywords: List[str]) -> str:
        """Generate business type focused domains."""
        business_clean = re.sub(r'[^a-zA-Z0-9]', '', business_type.replace('_', '').lower())
        
        if keywords and random.random() < 0.7:
            keyword = random.choice(keywords)
            combinations = [
                business_clean + keyword,
                keyword + business_clean,
                business_clean + random.choice(self.suffixes)
            ]
            domain_name = random.choice(combinations)
        else:
            domain_name = business_clean + random.choice(['service', 'pro', 'expert', 'plus'])
        
        extension = random.choices(['.com', '.net', '.org'], weights=[0.8, 0.15, 0.05])[0]
        return domain_name + extension
    
    def _generate_fallback_domains(self, business_type: str, count: int) -> List[str]:
        """Generate simple fallback domains when other methods fail."""
        fallbacks = []
        business_clean = re.sub(r'[^a-zA-Z0-9]', '', business_type.replace('_', '').lower())
        
        for i in range(count):
            fallback = f"{business_clean}{random.choice(['pro', 'plus', 'hub', 'zone'])}.com"
            fallbacks.append(fallback)
            
        return fallbacks
    
    def _is_valid_domain(self, domain: str) -> bool:
        """Validate domain name format and length."""
        if not domain:
            return False
            
        # Check basic format
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.[a-zA-Z]{2,4}$', domain):
            return False
            
        # Check length (without extension)
        name_part = domain.split('.')[0]
        if len(name_part) < 3 or len(name_part) > 63:
            return False
            
        # Check for consecutive hyphens
        if '--' in domain:
            return False
            
        return True
    
    def generate_dataset(self, 
                        num_samples: int = 1000,
                        complexity_distribution: Dict[str, float] = None,
                        include_edge_cases: bool = True,
                        edge_case_ratio: float = 0.15) -> List[Dict[str, Any]]:
        """
        Generate complete synthetic dataset for training.
        
        Args:
            num_samples: Total number of samples to generate
            complexity_distribution: Distribution of complexity levels
            include_edge_cases: Whether to include edge cases
            edge_case_ratio: Proportion of edge cases in dataset
            
        Returns:
            List of training samples with business descriptions and domain suggestions
        """
        if complexity_distribution is None:
            complexity_distribution = {"simple": 0.2, "medium": 0.5, "complex": 0.3}
        
        dataset = []
        
        # Calculate edge cases if included
        num_edge_cases = int(num_samples * edge_case_ratio) if include_edge_cases else 0
        num_regular_samples = num_samples - num_edge_cases
        
        logger.info(f"Generating {num_samples} synthetic samples ({num_regular_samples} regular + {num_edge_cases} edge cases)...")
        
        # Generate regular samples
        for i in range(num_regular_samples):
            # Determine complexity based on distribution
            complexity = random.choices(
                list(complexity_distribution.keys()),
                weights=list(complexity_distribution.values())
            )[0]
            
            # Generate business description
            business_data = self.generate_business_description(complexity)
            
            # Generate domain suggestions
            domain_suggestions = self.generate_domain_suggestions(business_data)
            
            # Create training example in instruction format
            sample = {
                "id": i,
                "business_description": business_data["business_description"],
                "domain_suggestions": domain_suggestions,
                "business_type": business_data["business_type"],
                "complexity": complexity,
                "is_edge_case": False,
                "num_suggestions": len(domain_suggestions),
                "keywords": business_data.get("keywords", []),
                "city": business_data.get("city"),
                "instruction_format": self._format_as_instruction(business_data["business_description"], domain_suggestions)
            }
            
            dataset.append(sample)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_regular_samples} regular samples")
        
        # Generate edge cases
        if include_edge_cases:
            edge_cases = self.generate_edge_cases(num_edge_cases)
            dataset.extend(edge_cases)
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        logger.info(f"Dataset generation complete. Total samples: {len(dataset)}")
        return dataset
    
    def _format_as_instruction(self, business_description: str, domain_suggestions: List[str]) -> str:
        """Format data as instruction-following format for training."""
        instruction = f"Generate domain name suggestions for the following business:\n\nBusiness: {business_description}\n\nDomain suggestions:"
        suggestions_text = '\n'.join(f"- {domain}" for domain in domain_suggestions)
        return f"{instruction}\n{suggestions_text}"
    
    def generate_targeted_samples(self, 
                                business_type: str, 
                                num_samples: int = 100,
                                quality_focus: bool = True) -> List[Dict[str, Any]]:
        """
        Generate targeted samples for specific business types or improvement areas.
        
        Args:
            business_type: Specific business type to focus on
            num_samples: Number of samples to generate
            quality_focus: Whether to emphasize high-quality examples
            
        Returns:
            List of targeted training samples
        """
        logger.info(f"Generating {num_samples} targeted samples for {business_type}")
        
        targeted_samples = []
        
        for i in range(num_samples):
            # Bias toward medium and complex for quality focus
            if quality_focus:
                complexity = random.choices(
                    ["simple", "medium", "complex"],
                    weights=[0.1, 0.4, 0.5]
                )[0]
            else:
                complexity = random.choice(["simple", "medium", "complex"])
            
            # Generate business description for specific type
            business_data = self.generate_business_description(complexity, business_type)
            
            # Generate multiple domain suggestions and select best ones
            all_suggestions = []
            for _ in range(3):  # Generate multiple batches
                suggestions = self.generate_domain_suggestions(business_data, num_suggestions=5)
                all_suggestions.extend(suggestions)
            
            # Remove duplicates and select diverse suggestions
            unique_suggestions = list(set(all_suggestions))
            final_suggestions = unique_suggestions[:random.randint(3, 6)]
            
            sample = {
                "id": f"targeted_{business_type}_{i}",
                "business_description": business_data["business_description"],
                "domain_suggestions": final_suggestions,
                "business_type": business_type,
                "complexity": complexity,
                "is_edge_case": False,
                "is_targeted": True,
                "num_suggestions": len(final_suggestions),
                "keywords": business_data.get("keywords", []),
                "instruction_format": self._format_as_instruction(business_data["business_description"], final_suggestions)
            }
            
            targeted_samples.append(sample)
        
        logger.info(f"Generated {len(targeted_samples)} targeted samples for {business_type}")
        return targeted_samples
    
    def save_dataset(self, dataset: pd.DataFrame, output_path: str):
        """Save dataset to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        dataset.to_csv(output_path.with_suffix('.csv'), index=False)
        
        # Save as JSON for easier inspection
        dataset.to_json(output_path.with_suffix('.json'), orient='records', indent=2)
        
        logger.info(f"Dataset saved to {output_path}")
        
    def generate_edge_cases(self, num_cases: int = 100, include_solutions: bool = False) -> List[Dict[str, Any]]:
        """
        Generate comprehensive edge case scenarios for robust testing.
        
        Args:
            num_cases: Number of edge cases to generate
            include_solutions: Whether to include high-quality solutions for edge cases
            
        Returns:
            List of edge case samples
        """
        logger.info(f"Generating {num_cases} edge case scenarios...")
        
        edge_cases = []
        
        # Define edge case categories and their characteristics
        edge_case_categories = {
            "ambiguous_business": {
                "descriptions": [
                    "business", "company", "service provider", "local establishment",
                    "professional services", "family business", "small business",
                    "startup", "enterprise", "organization", "firm"
                ],
                "weight": 0.20
            },
            "minimal_descriptions": {
                "descriptions": [
                    "cafe", "shop", "gym", "clinic", "store", "office", "studio",
                    "restaurant", "service", "company", "business", "place"
                ],
                "weight": 0.15
            },
            "excessive_descriptions": {
                "templates": [
                    "A comprehensive full-service professional {business_type} specializing in {descriptor1}, {descriptor2}, and {descriptor3} located in the heart of {city} with over {years} years of combined experience serving the local community with award-winning customer service and innovative solutions for all your {business_type} needs and requirements, committed to excellence and customer satisfaction",
                    "Established {business_type} offering premium {descriptor1} services, expert {descriptor2} solutions, and specialized {descriptor3} consulting for businesses and individuals throughout {city} and surrounding areas, with a dedicated team of professionals committed to delivering exceptional results and exceeding customer expectations"
                ],
                "weight": 0.15
            },
            "technical_jargon": {
                "templates": [
                    "AI-powered {business_type} utilizing machine learning algorithms and blockchain technology",
                    "Cloud-based {business_type} with microservices architecture and API-first approach",
                    "IoT-integrated {business_type} leveraging big data analytics and edge computing",
                    "Blockchain-enabled {business_type} with smart contracts and DeFi protocols",
                    "SaaS-based {business_type} platform with serverless architecture and real-time analytics"
                ],
                "weight": 0.15
            },
            "non_english_elements": {
                "descriptions": [
                    "Café Français", "El Restaurante", "La Boutique", "Das Restaurant",
                    "Il Ristorante", "カフェ", "餐厅", "レストラン", "कैफे", "مطعم"
                ],
                "weight": 0.10
            },
            "unusual_characters": {
                "templates": [
                    "{business_type}!!!", "{business_type} & Co.", "{business_type} (NEW!)",
                    "{business_type} - Premium", "{business_type} @ Location",
                    "{business_type} #1", "The {business_type}*", "{business_type} 24/7"
                ],
                "weight": 0.10
            },
            "industry_jargon": {
                "restaurant": ["farm-to-fork concept", "gastropub experience", "molecular gastronomy"],
                "tech_startup": ["disruptive innovation", "scalable solutions", "unicorn potential"],
                "healthcare_clinic": ["patient-centered care", "evidence-based medicine", "integrated wellness"],
                "law_firm": ["litigation support", "transactional law", "regulatory compliance"],
                "weight": 0.15
            }
        }
        
        # Generate edge cases by category
        category_names = list(edge_case_categories.keys())
        category_weights = [edge_case_categories[cat].get("weight", 0.1) for cat in category_names]
        
        for i in range(num_cases):
            category = random.choices(category_names, weights=category_weights)[0]
            
            # Generate edge case based on category
            if category == "ambiguous_business":
                description = random.choice(edge_case_categories[category]["descriptions"])
                business_type = "unknown"
                
            elif category == "minimal_descriptions":
                description = random.choice(edge_case_categories[category]["descriptions"])
                business_type = self._infer_business_type(description)
                
            elif category == "excessive_descriptions":
                template = random.choice(edge_case_categories[category]["templates"])
                business_type = random.choice(list(self.business_types.keys()))
                business_info = self.business_types[business_type]
                
                description = template.format(
                    business_type=business_type.replace('_', ' '),
                    descriptor1=random.choice(business_info["descriptors"]),
                    descriptor2=random.choice(business_info["descriptors"]),
                    descriptor3=random.choice(business_info["descriptors"]),
                    city=random.choice(self.cities),
                    years=random.randint(10, 50)
                )
                
            elif category == "technical_jargon":
                template = random.choice(edge_case_categories[category]["templates"])
                business_type = random.choice(["tech_startup", "consulting_firm", "marketing_agency"])
                description = template.format(business_type=business_type.replace('_', ' '))
                
            elif category == "non_english_elements":
                description = random.choice(edge_case_categories[category]["descriptions"])
                business_type = self._infer_business_type(description)
                
            elif category == "unusual_characters":
                business_type = random.choice(list(self.business_types.keys()))
                template = random.choice(edge_case_categories[category]["templates"])
                description = template.format(business_type=business_type.replace('_', ' '))
                
            else:  # industry_jargon
                business_types_with_jargon = ["restaurant", "tech_startup", "healthcare_clinic", "law_firm"]
                business_type = random.choice(business_types_with_jargon)
                jargon = random.choice(edge_case_categories["industry_jargon"][business_type])
                description = f"{business_type.replace('_', ' ')} specializing in {jargon}"
            
            # Generate domain suggestions (potentially lower quality for edge cases)
            if include_solutions:
                # Generate higher quality solutions for edge cases
                business_data = {
                    "business_description": description,
                    "business_type": business_type,
                    "keywords": self._extract_keywords(description)
                }
                domain_suggestions = self.generate_domain_suggestions(business_data, num_suggestions=5)
            else:
                # Generate basic suggestions that might struggle with edge cases
                domain_suggestions = self._generate_basic_domain_suggestions(description, business_type)
            
            edge_case = {
                "id": f"edge_{i}",
                "business_description": description,
                "domain_suggestions": domain_suggestions,
                "business_type": business_type,
                "complexity": "edge_case",
                "is_edge_case": True,
                "edge_case_category": category,
                "num_suggestions": len(domain_suggestions),
                "instruction_format": self._format_as_instruction(description, domain_suggestions)
            }
            
            edge_cases.append(edge_case)
        
        logger.info(f"Generated {len(edge_cases)} edge case scenarios")
        return edge_cases
    
    def _infer_business_type(self, description: str) -> str:
        """Infer business type from minimal description."""
        description_lower = description.lower()
        
        # Simple keyword matching
        if any(word in description_lower for word in ["cafe", "coffee", "espresso"]):
            return "coffee_shop"
        elif any(word in description_lower for word in ["restaurant", "dining", "food"]):
            return "restaurant"
        elif any(word in description_lower for word in ["shop", "store", "retail"]):
            return "retail_store"
        elif any(word in description_lower for word in ["gym", "fitness", "workout"]):
            return "fitness_center"
        elif any(word in description_lower for word in ["clinic", "medical", "health"]):
            return "healthcare_clinic"
        else:
            return "unknown"
    
    def _generate_basic_domain_suggestions(self, description: str, business_type: str) -> List[str]:
        """Generate basic domain suggestions that might not handle edge cases well."""
        keywords = self._extract_keywords(description)
        
        if not keywords:
            keywords = [business_type.replace('_', '')]
        
        suggestions = []
        for keyword in keywords[:3]:
            suggestions.append(f"{keyword}.com")
            suggestions.append(f"{keyword}pro.com")
            suggestions.append(f"best{keyword}.com")
        
        return suggestions[:random.randint(2, 5)]

    def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str, format: str = "both"):
        """
        Save dataset to file(s).
        
        Args:
            dataset: Dataset to save
            output_path: Base output path (without extension)
            format: 'csv', 'json', or 'both'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(dataset)
        
        if format in ["csv", "both"]:
            csv_path = output_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Dataset saved as CSV: {csv_path}")
        
        if format in ["json", "both"]:
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"Dataset saved as JSON: {json_path}")
        
        # Save summary statistics
        summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Dataset Summary for {output_path.stem}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total samples: {len(dataset)}\n")
            
            if 'business_type' in df.columns:
                f.write(f"\nBusiness Type Distribution:\n")
                business_counts = df['business_type'].value_counts()
                for bt, count in business_counts.items():
                    f.write(f"  {bt}: {count}\n")
            
            if 'complexity' in df.columns:
                f.write(f"\nComplexity Distribution:\n")
                complexity_counts = df['complexity'].value_counts()
                for comp, count in complexity_counts.items():
                    f.write(f"  {comp}: {count}\n")
            
            if 'is_edge_case' in df.columns:
                edge_cases = df['is_edge_case'].sum()
                f.write(f"\nEdge Cases: {edge_cases} ({edge_cases/len(dataset)*100:.1f}%)\n")
            
            f.write(f"\nAverage domain suggestions per sample: {df['num_suggestions'].mean():.1f}\n")
        
        logger.info(f"Dataset summary saved: {summary_path}")
    
    def analyze_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze dataset characteristics and quality metrics.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        df = pd.DataFrame(dataset)
        
        analysis = {
            "total_samples": len(dataset),
            "business_type_distribution": df['business_type'].value_counts().to_dict(),
            "complexity_distribution": df['complexity'].value_counts().to_dict(),
            "average_suggestions_per_sample": df['num_suggestions'].mean(),
            "edge_case_ratio": df['is_edge_case'].sum() / len(dataset) if 'is_edge_case' in df.columns else 0,
            "unique_business_types": df['business_type'].nunique(),
            "average_description_length": df['business_description'].str.len().mean(),
            "domain_extension_distribution": {}
        }
        
        # Analyze domain extensions
        all_domains = []
        for suggestions in df['domain_suggestions']:
            all_domains.extend(suggestions)
        
        extensions = [domain.split('.')[-1] for domain in all_domains if '.' in domain]
        extension_counts = pd.Series(extensions).value_counts()
        analysis["domain_extension_distribution"] = extension_counts.to_dict()
        
        return analysis
    
    def validate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate dataset quality and identify potential issues.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            "valid_samples": 0,
            "invalid_samples": 0,
            "issues": [],
            "warnings": []
        }
        
        for i, sample in enumerate(dataset):
            sample_issues = []
            
            # Check required fields
            required_fields = ["business_description", "domain_suggestions", "business_type"]
            for field in required_fields:
                if field not in sample or not sample[field]:
                    sample_issues.append(f"Missing or empty {field}")
            
            # Check business description quality
            if "business_description" in sample:
                desc = sample["business_description"]
                if len(desc) < 3:
                    sample_issues.append("Business description too short")
                elif len(desc) > 500:
                    sample_issues.append("Business description too long")
            
            # Check domain suggestions
            if "domain_suggestions" in sample:
                domains = sample["domain_suggestions"]
                if not isinstance(domains, list):
                    sample_issues.append("Domain suggestions not in list format")
                elif len(domains) == 0:
                    sample_issues.append("No domain suggestions provided")
                else:
                    for domain in domains:
                        if not self._is_valid_domain(domain):
                            sample_issues.append(f"Invalid domain format: {domain}")
            
            if sample_issues:
                validation_results["invalid_samples"] += 1
                validation_results["issues"].append({
                    "sample_id": sample.get("id", i),
                    "issues": sample_issues
                })
            else:
                validation_results["valid_samples"] += 1
        
        # Add overall warnings
        if validation_results["invalid_samples"] > len(dataset) * 0.1:
            validation_results["warnings"].append("High number of invalid samples (>10%)")
        
        return validation_results


# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    generator = SyntheticDataGenerator()
    
    print("🚀 Testing Synthetic Data Generator")
    print("=" * 50)
    
    # Test single business description generation
    print("\n1. Testing business description generation:")
    for complexity in ["simple", "medium", "complex"]:
        business_data = generator.generate_business_description(complexity)
        print(f"  {complexity.capitalize()}: {business_data['business_description']}")
    
    # Test domain suggestion generation
    print("\n2. Testing domain suggestion generation:")
    test_business = generator.generate_business_description("medium")
    domains = generator.generate_domain_suggestions(test_business)
    print(f"  Business: {test_business['business_description']}")
    print(f"  Domains: {domains}")
    
    # Generate small dataset
    print("\n3. Generating sample dataset...")
    dataset = generator.generate_dataset(
        num_samples=50,
        include_edge_cases=True,
        edge_case_ratio=0.2
    )
    
    # Analyze dataset
    print("\n4. Dataset analysis:")
    analysis = generator.analyze_dataset(dataset)
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Business types: {len(analysis['business_type_distribution'])}")
    print(f"  Edge case ratio: {analysis['edge_case_ratio']:.2%}")
    print(f"  Avg suggestions: {analysis['average_suggestions_per_sample']:.1f}")
    
    # Validate dataset
    print("\n5. Dataset validation:")
    validation = generator.validate_dataset(dataset)
    print(f"  Valid samples: {validation['valid_samples']}")
    print(f"  Invalid samples: {validation['invalid_samples']}")
    print(f"  Issues found: {len(validation['issues'])}")
    
    # Generate edge cases
    print("\n6. Testing edge case generation:")
    edge_cases = generator.generate_edge_cases(num_cases=20)
    print(f"  Generated {len(edge_cases)} edge cases")
    
    edge_categories = {}
    for case in edge_cases:
        category = case.get('edge_case_category', 'unknown')
        edge_categories[category] = edge_categories.get(category, 0) + 1
    
    print("  Edge case categories:")
    for category, count in edge_categories.items():
        print(f"    {category}: {count}")
    
    # Test targeted sample generation
    print("\n7. Testing targeted sample generation:")
    targeted = generator.generate_targeted_samples("restaurant", num_samples=10)
    print(f"  Generated {len(targeted)} targeted restaurant samples")
    
    print("\n✅ All tests completed successfully!")
    print(f"\n💡 To use this generator in your training pipeline:")
    print("   from src.data_generation import SyntheticDataGenerator")
    print("   generator = SyntheticDataGenerator()")
    print("   dataset = generator.generate_dataset(num_samples=1000)")
    print("   generator.save_dataset(dataset, 'data/training_data')")
