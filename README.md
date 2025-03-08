This is the full code for frontend and backend of vaxchat.
To run vaxchat.py you must have .env file named with the following properties (hopefully python version is not conflict):
  langchain.env:
    LANGCHAIN_TRACING 
    LANGCHAIN_API_KEY
    LANGCHAIN_ENDPOINT
  azure.env:
    OPEN_AI_KEY
    AZURE_ENDPOINT
    OPEN_AI_ORG
  neo4j.env
    NEO4J_URI
    NEO4J_USER
    NEO4J_PASSWORD

For the frontend, it uses npm and vite, hopefully there is no version/dependecy conflicts 🤞
