import os
import json
import requests
from typing import Dict, Any, Optional
import anthropic  # For Anthropic's Claude
import openai  # For OpenAI's models
import google.generativeai as genai  # For Google's models

class AIPromptService:
    def __init__(self, config_path: str = 'ai_config.json'):
        """
        Initialize the AI Prompt Service with configuration management.
        
        :param config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._validate_config()
        self._initialize_clients()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        :param config_path: Path to configuration file
        :return: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}. Using default configuration.")
            return self._create_default_config()
        except json.JSONDecodeError:
            print(f"Invalid JSON in config file at {config_path}. Using default configuration.")
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create a default configuration if no config file exists.
        
        :return: Default configuration dictionary
        """
        return {
            "providers": {
                "anthropic": {
                    "api_key": "",
                    "model": "claude-3-5-sonnet-20240620",
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "openai": {
                    "api_key": "",
                    "model": "gpt-4-turbo",
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "google": {
                    "api_key": "",
                    "model": "gemini-pro",
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            }
        }

    def _validate_config(self):
        """
        Validate the configuration dictionary.
        """
        required_keys = ['providers']
        provider_keys = ['anthropic', 'openai', 'google']
        
        if not all(key in self.config for key in required_keys):
            raise ValueError("Invalid configuration: Missing required keys")
        
        providers = self.config['providers']
        for provider in provider_keys:
            if provider not in providers:
                print(f"Warning: {provider} configuration not found in config")

    def _initialize_clients(self):
        """
        Initialize API clients for different providers.
        """
        # Anthropic Client
        anthropic_config = self.config['providers'].get('anthropic', {})
        if anthropic_config.get('api_key'):
            self.anthropic_client = anthropic.Anthropic(
                api_key=anthropic_config['api_key']
            )
        else:
            self.anthropic_client = None

        # OpenAI Client
        openai_config = self.config['providers'].get('openai', {})
        if openai_config.get('api_key'):
            openai.api_key = openai_config['api_key']
        else:
            openai.api_key = None

        # Google Client
        google_config = self.config['providers'].get('google', {})
        if google_config.get('api_key'):
            genai.configure(api_key=google_config['api_key'])
        else:
            genai = None
    #def process_prompts(self, prompt_data: Dict[str,str]) -> List[Dict[str, Any]]:
    def process_prompts(self, input_file, output_file=None):
        """
        Process prompts from input file and generate AI responses
        
        :param input_file: Path to input file with prompts
        :param output_file: Optional path to save responses
        """
        # Read prompts from file
        try:
            with aiofiles.open(input_file, 'r') as f:
                prompts = json.load(file)
        except FileNotFoundError:
            print(f"Error: Input file {input_file} not found.")
            return
        
        # Remove any whitespace and filter out empty lines
        #prompts = [prompt.strip() for prompt in prompts if prompt.strip()]
        
        # Process prompts
        results = []
        for prompt in prompts:
            enhanced_prompt = f"{prompt.get('prompt','')} Keywords: {', '.join(prompt.get('keywords',[])}"
            try:
                response = self.generate_response(enhanced_prompt)
                results.append({
                    'prompt': prompt,
                    'keywords': keywords,
                    'response': response
                })
                print(f"Processed prompt: {prompt[:50]}...")
            except Exception as e:
                print(f"Error processing prompt '{prompt}': {e}")
	        results.append({
                    "prompt": prompt,
                    "keywords": keywords,
                    "response": None,
                    "error": str(e)
                })	
        
        # Save results
        #if output_file:
        #    self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str): 
    #def save_results(self, results, output_file):
        """
        Save results to a JSON file
        
        :param results: List of processed prompts and responses
        :param output_file: Path to save output file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(results, file, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        except IOError:
            print(f"Error: Could not write to file {output_file}")
            #with aiofiles.open(output_file, 'w') as f:
            #    f.write(json.dumps(results, indent=2))
            #print(f"Results saved to {output_file}")
        #except Exception as e:
         #   print(f"Error saving results: {e}")
    
    def generate_response(
        self, 
        prompt: str, 
        provider: str = 'anthropic', 
        additional_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response from the specified AI provider.
        
        :param prompt: Input prompt for the AI
        :param provider: AI service provider (anthropic, openai, google)
        :param additional_params: Additional parameters to customize the request
        :return: Generated AI response
        """
        # Merge default and additional parameters
        default_params = self.config['providers'].get(provider, {})
        params = {**default_params, **(additional_params or {})}

        try:
            if provider == 'anthropic':
                return self._generate_anthropic_response(prompt, params)
            elif provider == 'openai':
                return self._generate_openai_response(prompt, params)
            elif provider == 'google':
                return self._generate_google_response(prompt, params)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            print(f"Error generating response from {provider}: {e}")
            return ""

    def _generate_anthropic_response(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Generate response using Anthropic's Claude model.
        
        :param prompt: Input prompt
        :param params: Configuration parameters
        :return: Generated response
        """
        if not self.anthropic_client:
            raise ValueError("Anthropic client not configured")

        response = self.anthropic_client.messages.create(
            model=params.get('model', 'claude-3-5-sonnet-20240620'),
            max_tokens=params.get('max_tokens', 1000),
            temperature=params.get('temperature', 0.7),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text

    def _generate_openai_response(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Generate response using OpenAI's models.
        
        :param prompt: Input prompt
        :param params: Configuration parameters
        :return: Generated response
        """
        if not openai.api_key:
            raise ValueError("OpenAI client not configured")

        response = openai.ChatCompletion.create(
            model=params.get('model', 'gpt-4-turbo'),
            max_tokens=params.get('max_tokens', 1000),
            temperature=params.get('temperature', 0.7),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def _generate_google_response(self, prompt: str, params: Dict[str, Any]) -> str:
        """
        Generate response using Google's Gemini model.
        
        :param prompt: Input prompt
        :param params: Configuration parameters
        :return: Generated response
        """
        if not genai:
            raise ValueError("Google client not configured")

        model = genai.GenerativeModel(
            params.get('model', 'gemini-pro')
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=params.get('max_tokens', 1000),
                temperature=params.get('temperature', 0.7)
            )
        )
        return response.text

def main():
    
    processor = AIPromptProcessor()
    
    try:
        # Read prompts from file
        prompts_file = 'prompts.txt'
        prompts = processor.read_prompts_from_file(prompts_file)
        keywords = ['technology', 'innovation', 'AI']
        results = processor.process_prompts(prompts, keywords)
        processor.save_results(results, 'ai_responses.json')
        #processor.process_prompts(
            input_file='prompts.json', 
            output_file='responses.json'
        )
    #except Exception as e:
    #    print(f"An error occurred: {e}")
    

if __name__ == '__main__':
    main()