
import os
from dotenv import load_dotenv

print("--- Running test to check for HF_TOKEN ---")

# This is the same logic used in main.py to find the .env file.
# It looks for a file named '.env' in the same directory as this script.
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

print(f"Attempting to load environment variables from: {dotenv_path}")
load_dotenv(dotenv_path=dotenv_path)

# Now, try to read the HF_TOKEN
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    print("\nSUCCESS: The HF_TOKEN was loaded successfully from the .env file.")
    # We will only print the first few characters to confirm without exposing the secret.
    print(f"Token value starts with: '{hf_token[:4]}...'")
else:
    print("\nFAILURE: The HF_TOKEN could NOT be loaded.")
    print("Please double-check the following:")
    print("1. Ensure the file 'backend/.env' exists.")
    print("2. Ensure the file contains exactly: HF_TOKEN='your_real_token_here'")

print("\n--- Test complete ---")
