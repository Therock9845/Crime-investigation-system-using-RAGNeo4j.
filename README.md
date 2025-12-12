# Crime-investigation-system-using-RAGNeo4j

An AI-powered Crime Investigation & Analysis System that integrates Retrieval-Augmented Generation (RAG), Graph-based knowledge representation using Neo4j, and natural-language querying to assist investigators in exploring crime data, connections, entities, and evidence relationships.

This system lets investigators:

Query crime data in natural language using LLMs

Visualize relationships between suspects, victims, locations, times, and evidence

Use Neo4j graph database for crime-network insights

Upload documents and extract structured knowledge

Run RAG over case files for accurate context-driven answers

# Tech Stack
Component	Technology

Frontend	Streamlit

LLM / RAG	OpenAI / Local LLM + Vector DB

Graph Database	Neo4j

Backend Logic	Python

Data Processing	LangChain / Custom pipelines

Visualization	Streamlit components + Neo4j Browser

# Features

Graph-based crime relationship modeling

Intelligent Q&A over case documents (RAG)

Crime-network visualization

Entity extraction (suspects, locations, evidence, timestamps…)

Document ingestion pipeline

Interactive Streamlit dashboard

# Installation & Setup Guide

Follow these steps to set up and run the project.

1️⃣ Clone the Repository
git clone https://github.com/Therock9845/Crime-investigation-system-using-RAGNeo4j..git
cd crime-investigation-system-using-RAGNeo4j

2️⃣ Install Python Dependencies

Ensure Python 3.10+ is installed.

Install all required packages:

pip install -r requirements.txt

3️⃣ Install Neo4j Desktop or Neo4j Server

Download Neo4j from the official website:
https://neo4j.com/download/

Then:

Create a new Database in neo4j

Start the database

Note your connection URL and username

Set a password for the database

Default Neo4j URI:

bolt://localhost:7687

4️⃣ Configure Environment Variables(very important)

Create a .env file in the project root or replace values in the existing .env:

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL

4️⃣LOCAL USAGE:

Install ollama  model in your system locally.

replace the LLM Provider to ollama instead of OpenAI and  do the ollama configuration.

⚠️ Important

Open Ai key is paid and is needed to run this project.If you want to just try the project as a trial you can use openrouters api key instead.

User must manually enter the password for their Neo4j instance

5️⃣ Run the Streamlit Application

Launch the main interface:

streamlit run app.py


This will open the Crime Investigation System in your browser
