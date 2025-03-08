This repository contains the full code for both the frontend and backend of VaxChat.

### To run `vaxchat.py`, create the following `.env` files with the respective properties:

#### `langchain.env`:
LANGCHAIN_TRACING=<your_value>  
LANGCHAIN_API_KEY=<your_value>  
LANGCHAIN_ENDPOINT=<your_value>

#### `azure.env`:
OPEN_AI_KEY=<your_value>  
AZURE_ENDPOINT=<your_value>  
OPEN_AI_ORG=<your_value>

#### `neo4j.env`:
NEO4J_URI=<your_value>  
NEO4J_USER=<your_value>  
NEO4J_PASSWORD=<your_value>

Make sure that there are no conflicts with the Python version you're using.

### Frontend setup:
The frontend uses `npm` and `Vite`. Ensure there are no version or dependency conflicts.
