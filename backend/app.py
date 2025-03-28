from flask import Flask, request, jsonify
from flask_cors import CORS 
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import re
from langchain.schema import BaseMessage
from transformers import pipeline
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from langchain_ollama.llms import OllamaLLM

app = Flask(__name__)
CORS(app)

load_dotenv("langchain.env")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")


global llama_model
llama_model = OllamaLLM(model="llama3.2")
load_dotenv("neo4j.env")
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")
global driver
driver = GraphDatabase.driver(uri, auth=(user, password))

def user_query_to_cypher(user_query):
    restructured_cypher_query = ChatPromptTemplate.from_messages([
        ("system", """
          You are a helpful assistant that rewrites natural language user queries into Cypher queries for a Neo4j vaccine database.

		Include nodes, relationships, and relevant properties in the queries.
		Use a default limit of 5 for specific queries, and a limit of 15 for abstract queries (e.g., "What is a description of Hepatitis C") only if the user does not specify a limit.
		Filter out empty or null values using IS NOT NULL, <> "", and <> "N/A".
		Keep in mind that all properties and node names, including numbers (e.g., HOST_ID), are treated as strings.
		Create robust and flexible queries when matching strings by using CONTAINS to ensure a match is found.
		Trim values to ensure better matching.
		Your response should only include the Cypher query itself.

		Node Types and Properties:
			HostName (name, HOST_ID, HOST_SCIENTIFIC_NAME, HOST_TAXONOMY_ID)
			PathogenName (name, PATHOGEN_ID, TAXON_ID, DISEASE_NAME, HOST_RANGE, ORGANISM_TYPE, PREPARATION_VO_ID, VACCINE_VO_ID, PROTEIN_VO_ID, PATHOGENESIS, PROTECTIVE_IMMUNITY, GRAM)
			VaccineName (name, VACCINE_ID, IS_COMBINATION_VACCINE, DESCRIPTION, ADJUVANT, STORAGE, VIRULENCE, PREPARATION, BRAND_NAME, FULL_TEXT, ANTIGEN, CURATION_FLAG, VECTOR, PROPER_NAME, MANUFACTURER, CONTRAINDICATION, STATUS, LOCATION_LICENSED, ROUTE, VO_ID, USAGE_AGE, MODEL_HOST, PRESERVATIVE, ALLERGEN, PREPARATION_VO_ID, HOST_SPECIES2, CVX_CODE, CVX_DESC)
		Relationships:
			(vn:VaccineName)-[:TARGETS_PATHOGEN]->(pn:PathogenName)
			(vn:VaccineName)-[:TARGETS_HOST]->(hn:HostName)

		Example Queries:
			User Query: 
				"Find vaccines targeting pathogens that cause disease in humans." 
			Cypher Query:
				MATCH (v:VaccineName)-[:TARGETS_PATHOGEN]->(p:PathogenName)
				MATCH (h:HostName {{name: "Human"}})
				WHERE p.DISEASE_NAME IS NOT NULL AND trim(p.DISEASE_NAME) <> ""
				RETURN v.name AS Vaccine, p.name AS Pathogen, p.DISEASE_NAME AS DISEASE
				LIMIT 5

			User Query: 
			"What vaccines were made by Pfizer Inc.?" 
			Cypher Query:
				MATCH (v:VaccineName)
				WHERE toLower(v.MANUFACTURER) CONTAINS toLower("Pfizer")
				RETURN v.name
				LIMIT 5

			User Query: 
				"What are the most common routes for Brucella vaccines?" 
			Cypher Query:
				MATCH (v:VaccineName)-[:TARGETS_PATHOGEN]->(p:PathogenName)
				WHERE toLower(p.name) CONTAINS toLower("Brucella")
				AND v.ROUTE IS NOT NULL
				AND toLower(trim(v.ROUTE)) <> "na"
				AND trim(v.ROUTE) <> ""
				RETURN v.ROUTE AS Route, COUNT(v.ROUTE) AS RouteCount
				ORDER BY RouteCount DESC
				LIMIT 5

			User Query: 
				"What are Influenza vaccines?" 
			Cypher Query:
				MATCH (v:VaccineName)-[:TARGETS_PATHOGEN]->(p:PathogenName)
				WHERE toLower(p.name) CONTAINS toLower("Influenza")
				AND v.DESCRIPTION IS NOT NULL
				AND toLower(v.DESCRIPTION) <> "na"
				AND v.DESCRIPTION <> ""
				RETURN v.DESCRIPTION AS VaccineDescription
				LIMIT 15

			User Query: 
				"Can you give me a list of 10 vaccine manufacturers by number of vaccines?" 
			Cypher Query:
				MATCH (v:VaccineName)
				WHERE v.MANUFACTURER IS NOT NULL AND v.MANUFACTURER <> ""
				WITH v.MANUFACTURER AS Manufacturer, COUNT(v) AS VaccineCount
				RETURN Manufacturer, VaccineCount
				ORDER BY VaccineCount DESC
				LIMIT 10
         """),
         ("system", "The user query to convert is: {user_query}")])
    
    
    llm_chain = restructured_cypher_query | llama_model
    converted_cypher_query = llm_chain.invoke({"user_query": user_query})
    cleaned_query = re.sub(r'```cypher\s*|\s*```', '', converted_cypher_query, flags=re.DOTALL)  # Removes backticks and `cypher`
    cleaned_query = re.search(r'(?i)(MATCH.*)', cleaned_query, flags=re.DOTALL).group(1) # Extracts everything starting from 'MATCH'
    #cleaned_query = re.search(r'(?s).*?LIMIT \d', cleaned_query, flags=re.DOTALL).group()
    return cleaned_query

def context_generating_answer(cypher_query, user_query):
    results = ''
    try: 
        with driver.session() as session:
            retrieved_data = session.run(cypher_query)
            results = retrieved_data.data()
            if results is None:
                print("No Results Found.")
    except Exception as e:
        print(f"\nERROR ~~~ {e} ~~~ ERROR\n")
    final_structured_output = ChatPromptTemplate([
        ("system", """
        You are a helpful assistant that answers user queries with retrieved data from a vaccine graph database Answer the question to the best
         of your ability keep answers short. If you lack confidence in your answer, indicate that the data retrieved may not be relevant to the user query.
        """),
        ("human", "The user query is {user_query} and the retrieved data: {retrieved_data}")
    ])
    try:
        llm_chain = final_structured_output | llama_model
        final_output = llm_chain.invoke({"user_query": user_query, "retrieved_data": results})
        return final_output, results
    except Exception as e:
        print(f"ERROR ~~~ {e} ~~~ ERROR")
    
@app.route('/api/chat', methods=['POST'])
def retrieveQuery():
    data = request.get_json()
    if not data or 'input' not in data:
        return jsonify({"error": "Invalid input"}, 400)
    user_query = data['input']
    cleaned_query = user_query_to_cypher(user_query)
    print(f"cleaned query: {cleaned_query}")
    answer, results = context_generating_answer(cleaned_query, user_query)
    print(f"answer: {answer}")
    bot_response = f"Processed query: {answer}"
    print(f"bot_response: {bot_response}")
    print(f"Results: {str(results)}")
    
    return jsonify({"response": bot_response, "cypher": cleaned_query, "data": str(results)})

if __name__ == '__main__':
    app.run(debug=True)