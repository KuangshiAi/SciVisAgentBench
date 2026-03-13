"""
Generic LLM-based evaluation helper using GPT-4o or Claude

This module provides a generic interface for evaluating scientific visualizations
using Large Language Models. It can be used by any test case evaluator.
Supports both OpenAI and Anthropic models.
"""
from openai import OpenAI
import base64
import os
import json
import re
from typing import List, Dict, Any

class LLMEvaluator:
    # OpenAI model pricing (per 1M tokens) as of September 2025
    # Anthropic model pricing (per 1M tokens) as of March 2026
    MODEL_PRICING = {
    # Anthropic Claude models
    "claude-opus-4.6": {"input": 15.00, "cached_input": 1.50, "output": 75.00, "provider": "anthropic"},
    "claude-opus-4-6": {"input": 15.00, "cached_input": 1.50, "output": 75.00, "provider": "anthropic"},
    "claude-sonnet-4.6": {"input": 3.00, "cached_input": 0.30, "output": 15.00, "provider": "anthropic"},
    "claude-sonnet-4-6": {"input": 3.00, "cached_input": 0.30, "output": 15.00, "provider": "anthropic"},
    "claude-sonnet-4.5": {"input": 3.00, "cached_input": 0.30, "output": 15.00, "provider": "anthropic"},
    "claude-sonnet-4-5": {"input": 3.00, "cached_input": 0.30, "output": 15.00, "provider": "anthropic"},
    "claude-haiku-4.5": {"input": 0.80, "cached_input": 0.08, "output": 4.00, "provider": "anthropic"},
    "claude-haiku-4-5": {"input": 0.80, "cached_input": 0.08, "output": 4.00, "provider": "anthropic"},

    # OpenAI models
    # GPT-5 series
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},

    # GPT-5 chat
    "gpt-5.2-chat-latest": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},

    # GPT-5 codex
    "gpt-5.1-codex-max": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.1-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.1-codex-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},

    # GPT-5 pro
    "gpt-5.2-pro": {"input": 21.00, "cached_input": None, "output": 168.00},
    "gpt-5-pro": {"input": 15.00, "cached_input": None, "output": 120.00},

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
    "gpt-realtime-mini": {"input": 0.60, "cached_input": 0.06, "output": 2.40},
    "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
    "gpt-4o-mini-realtime-preview": {"input": 0.60, "cached_input": 0.30, "output": 2.40},

    # Audio models
    "gpt-audio": {"input": 2.50, "cached_input": None, "output": 10.00},
    "gpt-audio-mini": {"input": 0.60, "cached_input": None, "output": 2.40},
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

    # Search & tools
    "gpt-5-search-api": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-4o-search-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
    "gpt-4o-mini-search-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
    "computer-use-preview": {"input": 3.00, "cached_input": None, "output": 12.00},

    # Image models
    "gpt-image-1.5": {"input": 5.00, "cached_input": 1.25, "output": 10.00},
    "chatgpt-image-latest": {"input": 5.00, "cached_input": 1.25, "output": 10.00},
    "gpt-image-1": {"input": 5.00, "cached_input": 1.25, "output": None},
    "gpt-image-1-mini": {"input": 2.00, "cached_input": 0.20, "output": None},

    # Codex
    "codex-mini-latest": {"input": 1.50, "cached_input": 0.375, "output": 6.00},

    # Legacy (unchanged)
    "gpt-4": {"input": 30.00, "cached_input": None, "output": 60.00},
    "gpt-4-turbo": {"input": 10.00, "cached_input": None, "output": 30.00},
    }
    
    def __init__(self, api_key=None, model="gpt-4o", max_tokens=1000, temperature=0.1, base_url=None):
        """
        Initialize the LLM evaluator

        Args:
            api_key (str): API key for OpenAI or Anthropic. If None, will try to get from environment
                          (OPENAI_API_KEY or ANTHROPIC_API_KEY)
            model (str): Model to use (OpenAI or Anthropic)
            max_tokens (int): Maximum tokens for response
            temperature (float): Temperature for response generation
            base_url (str): Optional custom API endpoint for OpenAI-compatible APIs.
                          Can be set via OPENAI_BASE_URL environment variable.
        """
        # Validate model
        if model not in self.MODEL_PRICING:
            available_models = list(self.MODEL_PRICING.keys())
            raise ValueError(f"Unsupported model '{model}'. Supported models: {', '.join(available_models)}")

        # Determine provider from model
        model_info = self.MODEL_PRICING[model]
        self.provider = model_info.get("provider", "openai")

        # Get API key based on provider
        if api_key is None:
            if self.provider == "anthropic":
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
            else:
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")

        # Create appropriate client
        if self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
                print(f"Using Anthropic model: {model}")
            except ImportError:
                raise ImportError("anthropic package not found. Install it with: pip install anthropic")
        else:
            # Create OpenAI client with optional custom base_url
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
                print(f"Using custom OpenAI endpoint: {base_url}")
            self.client = OpenAI(**client_kwargs)

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = base_url
    
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
            "provider": self.provider,
            "evaluator_version": "2.1.0",
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
    
    def _should_use_max_completion_tokens(self) -> bool:
        """
        Determine if the model requires max_completion_tokens instead of max_tokens.

        Newer OpenAI models (GPT-5.x, O-series) use max_completion_tokens.
        Older models (GPT-4.x, GPT-4o) use max_tokens.
        Anthropic models use max_tokens.

        Returns:
            bool: True if model uses max_completion_tokens, False if it uses max_tokens
        """
        if self.provider == "anthropic":
            return False

        # Models that use max_completion_tokens
        models_using_max_completion_tokens = [
            # GPT-5 series
            "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5-mini", "gpt-5-nano",
            "gpt-5.2-chat-latest", "gpt-5.1-chat-latest", "gpt-5-chat-latest",
            "gpt-5.1-codex-max", "gpt-5.1-codex", "gpt-5-codex", "gpt-5.1-codex-mini",
            "gpt-5.2-pro", "gpt-5-pro",
            # O-series models
            "o1", "o1-pro", "o1-mini",
            "o3", "o3-pro", "o3-mini", "o3-deep-research",
            "o4-mini", "o4-mini-deep-research",
        ]

        # Check if the model starts with any of these prefixes or is in the list
        for model_name in models_using_max_completion_tokens:
            if self.model.startswith(model_name):
                return True

        return False

    def encode_image(self, image_path: str) -> tuple:
        """
        Encode image to base64 string and detect format.
        Converts unsupported formats to PNG.

        Returns:
            tuple: (base64_string, mime_type)
        """
        from PIL import Image
        import io

        # Detect actual image format (not just extension)
        # Use PIL instead of imghdr (which was removed in Python 3.13)
        try:
            with Image.open(image_path) as img:
                img_type = img.format.lower() if img.format else None
        except Exception as e:
            print(f"  Warning: Could not detect image format for {image_path}: {e}")
            img_type = None

        # Supported formats by OpenAI: png, jpeg, gif, webp
        # Supported formats by Anthropic: png, jpeg, gif, webp
        supported_formats = {'png', 'jpeg', 'gif', 'webp'}

        if img_type in supported_formats:
            # Direct encoding for supported formats
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = f"image/{img_type}"
            return encoded, mime_type
        else:
            # Convert to PNG for unsupported formats (e.g., TGA, BMP, etc.)
            print(f"  Converting {img_type or 'unknown'} format to PNG for: {os.path.basename(image_path)}")
            img = Image.open(image_path)

            # Convert to RGB if necessary (remove alpha channel)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Save to bytes buffer as PNG
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            encoded = base64.b64encode(buffer.read()).decode('utf-8')
            return encoded, "image/png"

    def _call_llm(self, prompt: str, images: List[tuple]) -> str:
        """
        Call the appropriate LLM API (OpenAI or Anthropic) with images.

        Args:
            prompt: The text prompt
            images: List of (base64_string, mime_type) tuples

        Returns:
            str: The LLM response text
        """
        if self.provider == "anthropic":
            # Anthropic API format
            content = [{"type": "text", "text": prompt}]

            # Add images in Anthropic format
            for base64_image, mime_type in images:
                # Extract media type from mime_type (e.g., "image/png" -> "png")
                media_type = mime_type.split('/')[-1]
                if media_type == "jpg":
                    media_type = "jpeg"
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": base64_image
                    }
                })

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )

            # Extract text from response
            return response.content[0].text

        else:
            # OpenAI API format
            content = [{"type": "text", "text": prompt}]

            # Add images in OpenAI format
            for base64_image, mime_type in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high"
                    }
                })

            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "temperature": self.temperature
            }

            # Use max_completion_tokens for newer models, max_tokens for older models
            if self._should_use_max_completion_tokens():
                api_params["max_completion_tokens"] = self.max_tokens
            else:
                api_params["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**api_params)

            return response.choices[0].message.content
    
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

        # Prepare images
        images = []

        # Add ground truth images
        for img_path in ground_truth_images:
            if os.path.exists(img_path):
                encoded_image, mime_type = self.encode_image(img_path)
                images.append((encoded_image, mime_type))

        # Add result images
        for img_path in result_images:
            if os.path.exists(img_path):
                encoded_image, mime_type = self.encode_image(img_path)
                images.append((encoded_image, mime_type))

        try:
            print(f"Evaluating visualization quality with {self.provider} LLM ({self.model})...")
            evaluation_text = self._call_llm(evaluation_prompt, images)
            
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

        # Prepare images
        images = []

        # Add result images
        for img_path in result_images:
            if os.path.exists(img_path):
                encoded_image, mime_type = self.encode_image(img_path)
                images.append((encoded_image, mime_type))

        try:
            print(f"Evaluating visualization quality with {self.provider} LLM (result images only)...")
            evaluation_text = self._call_llm(evaluation_prompt, images)

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

    def evaluate_text(self, evaluation_prompt: str) -> Dict[str, Any]:
        """
        Text-based evaluation using LLM (no images)

        Args:
            evaluation_prompt (str): The complete evaluation prompt to send to LLM

        Returns:
            Dict: Evaluation results with scores and explanations
        """
        try:
            print(f"Evaluating with {self.provider} LLM ({self.model}) - text-based rubric...")
            # Call LLM with no images
            evaluation_text = self._call_llm(evaluation_prompt, [])

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
