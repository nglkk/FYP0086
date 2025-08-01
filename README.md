﻿# FYP0086 
Project Name: Ethical Hacking of Large Langugae Models: Exploring Vulnerabilities and Mitigation Techniques

## Features
- 3 attack modes: Prompt Injection, Prompt Leaking and Jailbreaking
- Supports 3 LLMs: GPT-4, DeepSeek and Mistral
- Streamlit interface

## Adding API Keys
1. Using .env file (Recommended)
    - UPDATE: The .env has already been created for your convenience 
    - Create file named "apikeys.env"
    - Your format should follow: 
    Openai_key = apikey
    Deepseek_key = apikey
    Mistral_key = apikey
    - Add this code start of attack.py and defence.py (after imports)

    ```python
    load_dotenv(dotenv_path = "apikeys.env")  # Load API keys from .env file

    openai_key = os.getenv("openai_key")
    deepseek_key = os.getenv("deepseek_key")
    mistral_key = os.getenv("mistral_key")
    ```

    - Replace "YOUR API KEY HERE" to "openai_key", "deepseek_key" and "mistral_key"
        - 3 api keys to add for attack.py (1 for each model)
        - 4 api keys to add for defence.py (2 for GPT-4, 1 for DeepSeek, 1 for Mistral)
    

3. Adding directly to code
    - Replace "YOUR API KEY HERE" to your own API keys for each model 
        - 3 api keys to add for attack.py (1 for each model)
        - 4 api keys to add for defence.py (2 for GPT-4, 1 for DeepSeek, 1 for Mistral)
