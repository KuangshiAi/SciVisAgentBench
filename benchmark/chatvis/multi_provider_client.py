#!/usr/bin/env python3
"""
Multi-Provider Client for ChatVis

This module provides a unified interface for different LLM providers (OpenAI, Anthropic, Hugging Face)
similar to how the TinyAgent handles multiple providers.
"""

import os
import json
from typing import Dict, List, Optional, Any, AsyncIterator
from pathlib import Path


class MultiProviderClient:
    """Unified client for multiple LLM providers."""
    
    def __init__(self, config_path: Optional[str] = None, provider: Optional[str] = None, 
                 model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the multi-provider client.
        
        Args:
            config_path: Path to configuration file
            provider: Provider name (openai, anthropic, hf/nebius)
            model: Model name
            api_key: API key for the provider
        """
        self.config = {}
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.client = None
        self.pricing_info = {}
        
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        self._initialize_client()
    
    def _load_config(self, config_path: str):
        """Load configuration from JSON file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Override with config values if not provided in constructor
        if not self.provider:
            self.provider = self.config.get('provider', 'openai')
        if not self.model:
            self.model = self.config.get('model', 'gpt-4o')
        if not self.api_key:
            self.api_key = self.config.get('api_key')
        
        # Load pricing info
        self.pricing_info = self.config.get('price', {})
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == 'openai':
            self._init_openai_client()
        elif self.provider == 'anthropic':
            self._init_anthropic_client()
        elif self.provider in ['hf', 'nebius', 'huggingface']:
            self._init_hf_client()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            # Get API key from config, parameter, or environment
            api_key = self.api_key or self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found")
            
            # Ensure proper base_url - default to standard OpenAI endpoint
            base_url = self.config.get('base_url', 'https://api.openai.com/v1')
            if base_url == 'https://api.openai.com':  # Fix common mistake
                base_url = 'https://api.openai.com/v1'
                
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install with: pip install openai")
    
    def _init_anthropic_client(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
            
            # Get API key from config, parameter, or environment
            api_key = self.api_key or self.config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key not found")
            
            base_url = self.config.get('base_url', 'https://api.anthropic.com')
            self.client = Anthropic(api_key=api_key, base_url=base_url)
            
        except ImportError:
            raise ImportError("Anthropic package not installed. Please install with: pip install anthropic")
    
    def _init_hf_client(self):
        """Initialize Hugging Face client."""
        try:
            from huggingface_hub import InferenceClient
            
            # Get API key from config, parameter, or environment
            api_key = self.api_key or self.config.get('api_key') or os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_API_KEY')
            
            # For HF, we might not need an API key for some models
            if self.provider == 'nebius':
                # Nebius is a specific provider through HF
                self.client = InferenceClient(
                    model=self.model,
                    token=api_key,
                    base_url=self.config.get('base_url')
                )
            else:
                self.client = InferenceClient(
                    model=self.model,
                    token=api_key,
                    base_url=self.config.get('base_url')
                )
                
        except ImportError:
            raise ImportError("Hugging Face Hub package not installed. Please install with: pip install huggingface_hub")
    
    def create_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Create a completion using the configured provider."""
        if self.provider == 'openai':
            return self._openai_completion(messages, **kwargs)
        elif self.provider == 'anthropic':
            return self._anthropic_completion(messages, **kwargs)
        elif self.provider in ['hf', 'nebius', 'huggingface']:
            return self._hf_completion(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _openai_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Create OpenAI completion."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
    
    def _anthropic_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Create Anthropic completion."""
        # Convert OpenAI format to Anthropic format
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                anthropic_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Anthropic requires alternating user/assistant messages
        # If first message is not from user, we need to handle it
        if anthropic_messages and anthropic_messages[0]['role'] != 'user':
            anthropic_messages.insert(0, {'role': 'user', 'content': 'Hello'})
        
        completion_kwargs = {
            'model': self.model,
            'messages': anthropic_messages,
            'max_tokens': kwargs.get('max_tokens', 4096),
        }
        
        if system_message:
            completion_kwargs['system'] = system_message
            
        return self.client.messages.create(**completion_kwargs)
    
    def _hf_completion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Create Hugging Face completion."""
        return self.client.chat_completion(
            messages=messages,
            model=self.model,
            **kwargs
        )
    
    def get_response_content(self, response: Any) -> str:
        """Extract content from response based on provider."""
        if self.provider == 'openai':
            return response.choices[0].message.content
        elif self.provider == 'anthropic':
            return response.content[0].text
        elif self.provider in ['hf', 'nebius', 'huggingface']:
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def get_token_usage(self, response: Any) -> Dict[str, int]:
        """Extract token usage from response based on provider."""
        if self.provider == 'openai':
            usage = response.usage
            return {
                'input_tokens': usage.prompt_tokens,
                'output_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens
            }
        elif self.provider == 'anthropic':
            usage = response.usage
            # Extract cache tokens if available (Anthropic prompt caching)
            cache_creation_tokens = getattr(usage, 'cache_creation_input_tokens', 0) or 0
            cache_read_tokens = getattr(usage, 'cache_read_input_tokens', 0) or 0

            return {
                'input_tokens': usage.input_tokens,
                'output_tokens': usage.output_tokens,
                'cache_creation_input_tokens': cache_creation_tokens,
                'cache_read_input_tokens': cache_read_tokens,
                'total_tokens': usage.input_tokens + usage.output_tokens + cache_creation_tokens + cache_read_tokens
            }
        elif self.provider in ['hf', 'nebius', 'huggingface']:
            # HF might not always provide detailed usage
            usage = getattr(response, 'usage', None)
            if usage:
                return {
                    'input_tokens': getattr(usage, 'prompt_tokens', 0),
                    'output_tokens': getattr(usage, 'completion_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0)
                }
            else:
                # Fallback - estimate tokens (rough approximation)
                content = self.get_response_content(response)
                estimated_tokens = len(content.split()) * 1.3  # Rough approximation
                return {
                    'input_tokens': 0,
                    'output_tokens': int(estimated_tokens),
                    'total_tokens': int(estimated_tokens)
                }
        else:
            return {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    
    def get_pricing_info(self) -> Dict[str, str]:
        """Get pricing information for cost calculation."""
        return self.pricing_info
    
    @classmethod
    def from_config_file(cls, config_path: str) -> 'MultiProviderClient':
        """Create client from configuration file."""
        return cls(config_path=config_path)


def load_chatvis_config(config_path: str) -> Dict[str, Any]:
    """Load ChatVis configuration from file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def create_client_from_config(config_path: str) -> MultiProviderClient:
    """Create a multi-provider client from configuration file."""
    return MultiProviderClient.from_config_file(config_path)