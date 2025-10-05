"""
Generic LLM-based evaluation helper using GPT-4o

This module provides a generic interface for evaluating scientific visualizations
using Large Language Models. It can be used by any test case evaluator.
"""
from openai import OpenAI
import base64
import os
import json
import re
from typing import List, Dict, Any

class LLMEvaluator:
    # OpenAI model pricing (per 1M tokens) as of September 2025
    MODEL_PRICING = {
        # GPT-5 series
        "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
        "gpt-5-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        
        # GPT-4.1 series
        "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
        
        # GPT-4o series
        "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-05-13": {"input": 5.00, "cached_input": None, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        
        # Realtime models
        "gpt-realtime": {"input": 4.00, "cached_input": 0.40, "output": 16.00},
        "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
        "gpt-4o-mini-realtime-preview": {"input": 0.60, "cached_input": 0.30, "output": 2.40},
        
        # Audio models
        "gpt-audio": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-audio-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-mini-audio-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
        
        # O-series models
        "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
        "o1-pro": {"input": 150.00, "cached_input": None, "output": 600.00},
        "o3-pro": {"input": 20.00, "cached_input": None, "output": 80.00},
        "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "o3-deep-research": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
        "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
        "o4-mini-deep-research": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "o1-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        
        # Special purpose models
        "codex-mini-latest": {"input": 1.50, "cached_input": 0.375, "output": 6.00},
        "gpt-4o-mini-search-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
        "gpt-4o-search-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
        "computer-use-preview": {"input": 3.00, "cached_input": None, "output": 12.00},
        "gpt-image-1": {"input": 5.00, "cached_input": 1.25, "output": None},
        
        # Legacy models (for backwards compatibility)
        "gpt-4": {"input": 30.00, "cached_input": None, "output": 60.00},
        "gpt-4-turbo": {"input": 10.00, "cached_input": None, "output": 30.00},
    }
    
    def __init__(self, api_key=None, model="gpt-4o", max_tokens=1000, temperature=0.1):
        """
        Initialize the LLM evaluator
        
        Args:
            api_key (str): OpenAI API key. If None, will try to get from environment
            model (str): OpenAI model to use
            max_tokens (int): Maximum tokens for response
            temperature (float): Temperature for response generation
        """
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Validate model
        if model not in self.MODEL_PRICING:
            available_models = list(self.MODEL_PRICING.keys())
            raise ValueError(f"Unsupported model '{model}'. Supported models: {', '.join(available_models)}")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def get_evaluator_info(self) -> Dict[str, Any]:
        """
        Get information about the evaluator configuration
        
        Returns:
            Dict: Evaluator metadata including model, settings, pricing, and version
        """
        pricing_info = self.MODEL_PRICING.get(self.model, {})
        
        return {
            "evaluator_type": "llm",
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "provider": "openai",
            "evaluator_version": "2.0.0",
            "pricing_per_1m_tokens": {
                "input": pricing_info.get("input"),
                "cached_input": pricing_info.get("cached_input"),
                "output": pricing_info.get("output"),
                "currency": "USD"
            }
        }
    
    def get_model_pricing(self, model: str = None) -> Dict[str, Any]:
        """
        Get pricing information for a specific model
        
        Args:
            model (str): Model name. If None, uses current model
            
        Returns:
            Dict: Pricing information for the model
        """
        target_model = model or self.model
        return self.MODEL_PRICING.get(target_model, {})
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """
        Get list of all supported models
        
        Returns:
            List[str]: List of supported model names
        """
        return list(cls.MODEL_PRICING.keys())
    
    @classmethod
    def get_model_categories(cls) -> Dict[str, List[str]]:
        """
        Get models organized by categories
        
        Returns:
            Dict: Models organized by category
        """
        categories = {
            "gpt5": [m for m in cls.MODEL_PRICING.keys() if m.startswith("gpt-5")],
            "gpt4.1": [m for m in cls.MODEL_PRICING.keys() if m.startswith("gpt-4.1")],
            "gpt4o": [m for m in cls.MODEL_PRICING.keys() if m.startswith("gpt-4o")],
            "gpt4_legacy": [m for m in cls.MODEL_PRICING.keys() if m.startswith("gpt-4") and not m.startswith("gpt-4o") and not m.startswith("gpt-4.1")],
            "realtime": [m for m in cls.MODEL_PRICING.keys() if "realtime" in m],
            "audio": [m for m in cls.MODEL_PRICING.keys() if "audio" in m and "realtime" not in m],
            "o_series": [m for m in cls.MODEL_PRICING.keys() if m.startswith("o")],
            "special": [m for m in cls.MODEL_PRICING.keys() if any(x in m for x in ["codex", "search", "computer-use", "image"])]
        }
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def evaluate_visualization(self, ground_truth_images: List[str], result_images: List[str], 
                             evaluation_prompt: str) -> Dict[str, Any]:
        """
        Generic visualization evaluation using LLM
        
        Args:
            ground_truth_images (List[str]): Paths to ground truth screenshots
            result_images (List[str]): Paths to result screenshots  
            evaluation_prompt (str): The complete evaluation prompt to send to LLM
            
        Returns:
            Dict: Evaluation results with scores and explanations
        """
        
        # Prepare image content for the API call
        image_content = []
        
        # Add ground truth images
        for i, img_path in enumerate(ground_truth_images):
            if os.path.exists(img_path):
                encoded_image = self.encode_image(img_path)
                image_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}",
                        "detail": "high"
                    }
                })
        
        # Add result images
        for i, img_path in enumerate(result_images):
            if os.path.exists(img_path):
                encoded_image = self.encode_image(img_path)
                image_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}",
                        "detail": "high"
                    }
                })

        try:
            print("Evaluating visualization quality with LLM...")
            if "gpt-5" in self.model:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": evaluation_prompt},
                                *image_content
                            ]
                        }
                    ]
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": evaluation_prompt},
                                *image_content
                            ]
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            
            # Parse the response
            evaluation_text = response.choices[0].message.content
            
            # Try multiple strategies to extract and parse JSON
            json_result = None
            
            # Strategy 1: Try to find JSON within markdown code blocks (```json ... ```)
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', evaluation_text, re.DOTALL)
            if code_block_match:
                try:
                    json_result = json.loads(code_block_match.group(1))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON from code block: {e}")
            
            # Strategy 2: Try to parse entire response as JSON
            if json_result is None:
                try:
                    json_result = json.loads(evaluation_text.strip())
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Try to find JSON object using improved regex (non-greedy, balanced)
            if json_result is None:
                # Find the first { and last } to extract JSON
                json_start = evaluation_text.find('{')
                json_end = evaluation_text.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    try:
                        json_str = evaluation_text[json_start:json_end+1]
                        json_result = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            if json_result:
                # Add evaluator metadata to the result
                json_result["evaluator_info"] = self.get_evaluator_info()
                return json_result
            
            # If all JSON parsing strategies fail, return a structured response with error
            print(f"Warning: Could not parse JSON from LLM response.")
            print(f"Response preview (first 500 chars): {evaluation_text[:500]}")
            return {
                "evaluation_text": evaluation_text,
                "error": "Could not parse JSON response",
                "raw_response": evaluation_text,
                "evaluator_info": self.get_evaluator_info()
            }
            
        except Exception as e:
            return {
                "error": f"LLM evaluation failed: {str(e)}",
                "raw_response": "",
                "evaluator_info": self.get_evaluator_info()
            }
    
    def evaluate_visualization_result_only(self, result_images: List[str], evaluation_prompt: str) -> Dict[str, Any]:
        """
        Visualization evaluation using LLM with result images only (no ground truth)
        
        Args:
            result_images (List[str]): Paths to result screenshots
            evaluation_prompt (str): The complete evaluation prompt to send to LLM
            
        Returns:
            Dict: Evaluation results with scores and explanations
        """
        
        # Prepare image content for the API call
        image_content = []
        
        # Add result images
        for i, img_path in enumerate(result_images):
            if os.path.exists(img_path):
                encoded_image = self.encode_image(img_path)
                image_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{encoded_image}",
                        "detail": "high"
                    }
                })

        try:
            print("Evaluating visualization quality with LLM (result images only)...")
            if "gpt-5" in self.model:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": evaluation_prompt},
                                *image_content
                            ]
                        }
                    ]
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": evaluation_prompt},
                                *image_content
                            ]
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            
            # Parse the response
            evaluation_text = response.choices[0].message.content
            
            # Try multiple strategies to extract and parse JSON
            json_result = None
            
            # Strategy 1: Try to find JSON within markdown code blocks (```json ... ```)
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', evaluation_text, re.DOTALL)
            if code_block_match:
                try:
                    json_result = json.loads(code_block_match.group(1))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON from code block: {e}")
            
            # Strategy 2: Try to parse entire response as JSON
            if json_result is None:
                try:
                    json_result = json.loads(evaluation_text.strip())
                except json.JSONDecodeError:
                    pass
            
            # Strategy 3: Try to find JSON object using improved regex (non-greedy, balanced)
            if json_result is None:
                # Find the first { and last } to extract JSON
                json_start = evaluation_text.find('{')
                json_end = evaluation_text.rfind('}')
                if json_start != -1 and json_end != -1 and json_end > json_start:
                    try:
                        json_str = evaluation_text[json_start:json_end+1]
                        json_result = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            if json_result:
                # Add evaluator metadata to the result
                json_result["evaluator_info"] = self.get_evaluator_info()
                return json_result
            
            # If all JSON parsing strategies fail, return a structured response with error
            print(f"Warning: Could not parse JSON from LLM response.")
            print(f"Response preview (first 500 chars): {evaluation_text[:500]}")
            return {
                "evaluation_text": evaluation_text,
                "error": "Could not parse JSON response",
                "raw_response": evaluation_text,
                "evaluator_info": self.get_evaluator_info()
            }
            
        except Exception as e:
            return {
                "error": f"LLM evaluation failed: {str(e)}",
                "raw_response": "",
                "evaluator_info": self.get_evaluator_info()
            }
