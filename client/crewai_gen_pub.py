import streamlit as st
from crewai import Agent, Task, Crew
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import json
import time
import ollama
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from streamlit_agraph import agraph, Node, Edge, Config
import io
import re
from anthropic import Anthropic
import google.generativeai as genai
from langchain_mistralai import ChatMistralAI

# Chargement des variables d'environnement
load_dotenv()

# Récupération des identifiants depuis .env
VALID_USERNAME = os.getenv("STREAMLIT_USERNAME")
VALID_PASSWORD = os.getenv("STREAMLIT_PASSWORD")

# Vérification des credentials (optionnel, car l'accès public ne nécessite pas de login)
if not VALID_USERNAME or not VALID_PASSWORD:
    st.warning("Les variables STREAMLIT_USERNAME et STREAMLIT_PASSWORD ne sont pas définies. Ollama nécessitera une connexion.")

class AgentGenerator:
    def __init__(self, provider: str = "openai", api_key: str = None):
        self.provider = provider
        self.api_key = api_key
        self.parameters = {
            "temperature": 0.7,
            "max_tokens": 1500,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        self.model = None
        self.model_id = self._set_default_model()

    def _set_default_model(self) -> str:
        defaults = {
            "openai": "gpt-4o-mini",
            "mistral": "mistral-small",
            "anthropic": "claude-3-sonnet",
            "google": "gemini-flash",
            "ollama": os.getenv("OLLAMA_MODEL")
        }
        return defaults.get(self.provider, "llama3.2")

    def _initialize_model(self):
        if self.model is None:
            if self.provider == "ollama":
                try:
                    ollama.list()
                    self.model = ollama
                except Exception as e:
                    st.error(f"Erreur : Ollama n'est pas détecté. Détails : {e}")
                    st.stop()
            elif self.provider == "openai":
                if not self.api_key:
                    st.error("Clé API OpenAI requise.")
                    st.stop()
                self.model = ChatOpenAI(api_key=self.api_key, model=self.model_id)
            elif self.provider == "mistral":
                if not self.api_key:
                    st.error("Clé API Mistral AI requise.")
                    st.stop()
                self.model = ChatMistralAI(api_key=self.api_key, model=self.model_id)
            elif self.provider == "anthropic":
                if not self.api_key:
                    st.error("Clé API Anthropic requise.")
                    st.stop()
                self.model = Anthropic(api_key=self.api_key)
            elif self.provider == "google":
                if not self.api_key:
                    st.error("Clé API Google requise.")
                    st.stop()
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_id)

    def analyze_prompt(self, user_prompt: str, framework: str) -> Dict[str, Any]:
        self._initialize_model()
        system_prompt = self._get_system_prompt_for_framework(framework)
        formatted_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            if self.provider == "ollama":
                response = self.model.generate(model=self.model_id, prompt=formatted_prompt, options=self.parameters)["response"]
            elif self.provider in ["openai", "mistral"]:
                response = self.model.invoke(formatted_prompt).content
            elif self.provider == "anthropic":
                response = self.model.messages.create(model=self.model_id, messages=[{"role": "user", "content": formatted_prompt}], max_tokens=1500).content[0].text
            elif self.provider == "google":
                response = self.model.generate_content(formatted_prompt).text
            
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                st.warning("Impossible d'extraire un JSON valide. Configuration par défaut utilisée.")
                return self._get_default_config(framework)
        except json.JSONDecodeError as e:
            st.error(f"Erreur de parsing JSON : {e}. Configuration par défaut utilisée.")
            return self._get_default_config(framework)
        except Exception as e:
            st.error(f"Erreur lors de l'analyse du prompt : {e}")
            return self._get_default_config(framework)

    def suggest_prompt_improvements(self, user_prompt: str) -> str:
        self._initialize_model()
        suggestion_prompt = f"""
        Analyze the following user prompt and suggest improvements to make it clearer, more specific, and better suited for generating AI agents:
        "{user_prompt}"
        Provide your suggestions in a concise paragraph.
        """
        try:
            if self.provider == "ollama":
                return self.model.generate(model=self.model_id, prompt=suggestion_prompt, options={"max_tokens": 200})["response"]
            elif self.provider in ["openai", "mistral"]:
                return self.model.invoke(suggestion_prompt).content
            elif self.provider == "anthropic":
                return self.model.messages.create(model=self.model_id, messages=[{"role": "user", "content": suggestion_prompt}], max_tokens=200).content[0].text
            elif self.provider == "google":
                return self.model.generate_content(suggestion_prompt).text
        except Exception as e:
            st.error(f"Erreur lors de la suggestion : {e}")
            return "Erreur lors de la génération des suggestions."

    def _get_system_prompt_for_framework(self, framework: str) -> str:
        if framework == "crewai":
            return """
            Vous êtes un expert dans la création d'assistants de recherche en IA utilisant CrewAI. En fonction de la demande de l'utilisateur,
            suggérez des agents appropriés, leurs rôles, outils et tâches. Formatez votre réponse en JSON avec cette structure :
            {
                "agents": [
                    {
                        "name": "nom de l'agent",
                        "role": "description spécifique du rôle",
                        "goal": "objectif clair",
                        "backstory": "histoire pertinente",
                        "tools": ["search", "wikipedia"],
                        "verbose": true,
                        "allow_delegation": true/false
                    }
                ],
                "tasks": [
                    {
                        "name": "nom de la tâche",
                        "description": "description détaillée",
                        "tools": ["outils requis"],
                        "agent": "nom de l'agent",
                        "expected_output": "résultat attendu spécifique"
                    }
                ]
            }
            """
        elif framework == "langgraph":
            return """
            Vous êtes un expert dans la création d'agents IA utilisant le framework LangGraph de LangChain. En fonction de la demande de l'utilisateur,
            suggérez des agents appropriés, leurs rôles, outils et nœuds pour le graphe. Formatez votre réponse en JSON avec cette structure :
            {
                "agents": [
                    {
                        "name": "nom de l'agent",
                        "role": "description spécifique du rôle",
                        "goal": "objectif clair",
                        "tools": ["search", "wikipedia"],
                        "llm": "llama3.2"
                    }
                ],
                "nodes": [
                    {
                        "name": "nom du nœud",
                        "description": "description détaillée",
                        "agent": "nom de l'agent"
                    }
                ],
                "edges": [
                    {
                        "source": "nom du nœud source",
                        "target": "nom du nœud cible",
                        "condition": "description de la condition (optionnelle)"
                    }
                ]
            }
            """
        elif framework == "react":
            return """
            Vous êtes un expert dans la création d'agents IA utilisant le framework ReAct. En fonction de la demande de l'utilisateur,
            suggérez des agents appropriés, leurs rôles, outils et étapes de raisonnement. Formatez votre réponse en JSON avec cette structure :
            {
                "agents": [
                    {
                        "name": "nom de l'agent",
                        "role": "description spécifique du rôle",
                        "goal": "objectif clair",
                        "tools": ["search", "wikipedia"],
                        "llm": "llama3.2"
                    }
                ],
                "tools": [
                    {
                        "name": "nom de l'outil",
                        "description": "description détaillée de ce que fait l'outil",
                        "parameters": {"param1": "description"}
                    }
                ],
                "examples": [
                    {
                        "query": "requête utilisateur exemple",
                        "thought": "processus de réflexion exemple",
                        "action": "action exemple",
                        "observation": "observation exemple",
                        "final_answer": "réponse finale exemple"
                    }
                ]
            }
            """
        return "Framework invalide"

    def _get_default_config(self, framework: str) -> Dict[str, Any]:
        if framework == "crewai":
            return {
                "agents": [{"name": "default_assistant", "role": "Assistant Général", "goal": "Aider avec les tâches", "backstory": "Assistant polyvalent", "tools": ["search"], "verbose": True, "allow_delegation": False}],
                "tasks": [{"name": "tache_basique", "description": "Gérer les requêtes de base", "tools": ["search"], "agent": "default_assistant", "expected_output": "Tâche terminée"}]
            }
        elif framework == "langgraph":
            return {
                "agents": [{"name": "default_assistant", "role": "Assistant Général", "goal": "Aider avec les tâches", "tools": ["search"], "llm": "llama3.2"}],
                "nodes": [{"name": "process_input", "description": "Traiter l'entrée utilisateur", "agent": "default_assistant"}],
                "edges": [{"source": "process_input", "target": "END", "condition": "tâche terminée"}]
            }
        elif framework == "react":
            return {
                "agents": [{"name": "default_assistant", "role": "Assistant Général", "goal": "Aider avec les tâches", "tools": ["search"], "llm": "llama3.2"}],
                "tools": [{"name": "search", "description": "Outil de recherche", "parameters": {"input": "requête"}}],
                "examples": [{"query": "Trouver des infos", "thought": "Recherche nécessaire", "action": "search", "observation": "Résultats trouvés", "final_answer": "Voici les infos"}]
            }
        return {}

def create_crewai_code(config: Dict[str, Any], provider: str, api_key: str) -> str:
    code = f"""from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
"""
    if provider == "openai":
        code += f"from langchain_openai import ChatOpenAI\nllm = ChatOpenAI(api_key='{api_key}', model='{config['agents'][0].get('llm', 'gpt-4')}')\n"
    elif provider == "mistral":
        code += f"from langchain_mistralai import ChatMistralAI\nllm = ChatMistralAI(api_key='{api_key}', model='{config['agents'][0].get('llm', 'mistral-large')}')\n"
    elif provider == "anthropic":
        code += f"from anthropic import Anthropic\nanthropic_client = Anthropic(api_key='{api_key}')\n"
    elif provider == "google":
        code += f"import google.generativeai as genai\ngenai.configure(api_key='{api_key}')\nmodel = genai.GenerativeModel('{config['agents'][0].get('llm', 'gemini-pro')}')\n"
    elif provider == "ollama":
        code += f"import ollama\n"

    code += """
search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

"""
    for agent in config["agents"]:
        role = repr(agent["role"])
        goal = repr(agent["goal"])
        backstory = repr(agent["backstory"])
        if provider in ["openai", "mistral"]:
            code += f"agent_{agent['name']} = Agent(\n    role={role},\n    goal={goal},\n    backstory={backstory},\n    verbose={agent['verbose']},\n    allow_delegation={agent['allow_delegation']},\n    tools=[search_tool, wiki_tool],\n    llm=llm\n)\n\n"
        elif provider == "anthropic":
            code += f"agent_{agent['name']} = Agent(\n    role={role},\n    goal={goal},\n    backstory={backstory},\n    verbose={agent['verbose']},\n    allow_delegation={agent['allow_delegation']},\n    tools=[search_tool, wiki_tool],\n    llm=anthropic_client\n)\n\n"  # Note : CrewAI nécessite une adaptation pour Anthropic
        elif provider == "google":
            code += f"agent_{agent['name']} = Agent(\n    role={role},\n    goal={goal},\n    backstory={backstory},\n    verbose={agent['verbose']},\n    allow_delegation={agent['allow_delegation']},\n    tools=[search_tool, wiki_tool],\n    llm=model\n)\n\n"  # Note : Adaptation nécessaire
        elif provider == "ollama":
            code += f"agent_{agent['name']} = Agent(\n    role={role},\n    goal={goal},\n    backstory={backstory},\n    verbose={agent['verbose']},\n    allow_delegation={agent['allow_delegation']},\n    tools=[search_tool, wiki_tool]\n)\n\n"  # Ollama local, pas de llm explicite
    
    for task in config["tasks"]:
        task_name = re.sub(r'\s+', '_', task["name"])
        description = repr(task["description"])
        expected_output = repr(task["expected_output"])
        # Check if 'agent' key exists, default to first agent if missing
        agent_name = task.get("agent", config["agents"][0]["name"] if config["agents"] else "default_assistant")
        code += "task_{} = Task(\n    description={},\n    agent=agent_{},\n    expected_output={}\n)\n\n".format(
            task_name, description, agent_name, expected_output
        )
    
    agents_list = ", ".join("agent_{}".format(a['name']) for a in config["agents"])
    tasks_list = ", ".join("task_{}".format(re.sub(r'\s+', '_', t['name'])) for t in config["tasks"])
    code += "crew = Crew(\n    agents=[{}],\n    tasks=[{}]\n)\nresult = crew.kickoff()\n\nprint(result)".format(agents_list, tasks_list)
    return code


def create_langgraph_code(config: Dict[str, Any], provider: str, api_key: str) -> str:
    code = """from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import Dict, List, Any, TypedDict

class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str

"""
    if provider == "openai":
        code += f"from langchain_openai import ChatOpenAI\nllm = ChatOpenAI(api_key='{api_key}', model='{config['agents'][0].get('llm', 'gpt-4')}')\n"
    elif provider == "mistral":
        code += f"from langchain_mistralai import ChatMistralAI\nllm = ChatMistralAI(api_key='{api_key}', model='{config['agents'][0].get('llm', 'mistral-large')}')\n"
    elif provider == "anthropic":
        code += f"from anthropic import Anthropic\nanthropic_client = Anthropic(api_key='{api_key}')\n"
    elif provider == "google":
        code += f"import google.generativeai as genai\ngenai.configure(api_key='{api_key}')\nmodel = genai.GenerativeModel('{config['agents'][0].get('llm', 'gemini-pro')}')\n"
    elif provider == "ollama":
        code += f"import ollama\n"

    code += """
search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [search_tool, wiki_tool]

"""
    for agent in config["agents"]:
        llm_model = agent.get('llm', 'llama3.2')
        if provider in ["openai", "mistral"]:
            code += f"def {agent['name']}_agent(state: AgentState) -> AgentState:\n    messages = state['messages']\n    response = llm.invoke(messages)\n    return {{\"messages\": messages + [response], \"next\": state.get(\"next\", \"\")}}\n\n"
        elif provider == "anthropic":
            code += f"def {agent['name']}_agent(state: AgentState) -> AgentState:\n    messages = state['messages']\n    response = anthropic_client.messages.create(model='{llm_model}', messages=[{{'role': 'user', 'content': messages[-1].content}}], max_tokens=1500).content[0].text\n    return {{\"messages\": messages + [AIMessage(content=response)], \"next\": state.get(\"next\", \"\")}}\n\n"
        elif provider == "google":
            code += f"def {agent['name']}_agent(state: AgentState) -> AgentState:\n    messages = state['messages']\n    response = model.generate_content(messages[-1].content).text\n    return {{\"messages\": messages + [AIMessage(content=response)], \"next\": state.get(\"next\", \"\")}}\n\n"
        elif provider == "ollama":
            code += f"def {agent['name']}_agent(state: AgentState) -> AgentState:\n    messages = state['messages']\n    response = ollama.generate(model='{llm_model}', prompt=messages[-1].content)['response']\n    return {{\"messages\": messages + [AIMessage(content=response)], \"next\": state.get(\"next\", \"\")}}\n\n"

    code += """def router(state: AgentState) -> str:
    return state.get("next", "END")

workflow = StateGraph(AgentState)
"""
    for node in config["nodes"]:
        code += f"workflow.add_node(\"{node['name']}\", {node['agent']}_agent)\n"
    for edge in config["edges"]:
        code += f"workflow.add_edge(\"{edge['source']}\", \"{edge['target']}\")\n" if edge["target"] != "END" else f"workflow.add_edge(\"{edge['source']}\", END)\n"
    if config["nodes"]:
        code += f"workflow.set_entry_point(\"{config['nodes'][0]['name']}\")\n"
    code += """
app = workflow.compile()

def run_agent(query: str) -> List[BaseMessage]:
    result = app.invoke({\"messages\": [HumanMessage(content=query)], \"next\": \"\"})
    return result[\"messages\"]

result = run_agent(\"Votre requête ici\")
for msg in result: print(f\"{msg.type}: {msg.content}\")
"""
    return code

def create_react_code(config: Dict[str, Any], provider: str, api_key: str) -> str:
    code = """from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [search_tool, wiki_tool]

"""
    if provider == "openai":
        code += f"from langchain_openai import ChatOpenAI\nllm = ChatOpenAI(api_key='{api_key}', model='{config['agents'][0].get('llm', 'gpt-4')}')\n"
    elif provider == "mistral":
        code += f"from langchain_mistralai import ChatMistralAI\nllm = ChatMistralAI(api_key='{api_key}', model='{config['agents'][0].get('llm', 'mistral-large')}')\n"
    elif provider == "anthropic":
        code += f"from anthropic import Anthropic\nanthropic_client = Anthropic(api_key='{api_key}')\n"
    elif provider == "google":
        code += f"import google.generativeai as genai\ngenai.configure(api_key='{api_key}')\nmodel = genai.GenerativeModel('{config['agents'][0].get('llm', 'gemini-pro')}')\n"
    elif provider == "ollama":
        code += f"import ollama\n"

    if config.get("agents"):
        agent = config["agents"][0]
        llm_model = agent.get('llm', 'llama3.2')
        role = repr(agent["role"])
        goal = repr(agent["goal"])
        if provider in ["openai", "mistral"]:
            code += f"""
react_prompt = ChatPromptTemplate.from_messages([
    ("system", \"\"\"Vous êtes {role}. Votre objectif est de {goal}.
Utilisez les outils : {{tool_descriptions}}
Format :
Question : La question de l'utilisateur
Thought : Que faire
Action : Une des {{tool_names}}
Action Input : Entrée de l'action
Observation : Résultat
... (répéter si nécessaire)
Thought : Je connais la réponse finale
Final Answer : Réponse finale\"\"\"), 
    ("human", "{{input}}")
])

agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent(query: str) -> str:
    return agent_executor.invoke({{"input": query}}).get("output", "Aucune réponse générée")

result = run_agent("Votre requête ici")
print(result)
"""
        # Note : Anthropic, Google, et Ollama nécessitent une adaptation supplémentaire pour ReAct
    return code

def create_code_block(config: Dict[str, Any], framework: str, provider: str, api_key: str) -> str:
    if framework == "crewai":
        return create_crewai_code(config, provider, api_key)
    elif framework == "langgraph":
        return create_langgraph_code(config, provider, api_key)
    elif framework == "react":
        return create_react_code(config, provider, api_key)
    return "# Framework invalide"

def display_graph(config: Dict[str, Any], framework: str):
    nodes, edges = [], []
    if framework == "crewai":
        for agent in config["agents"]:
            nodes.append(Node(id=agent["name"], label=agent["role"], size=25, color="#4CAF50"))
        for task in config["tasks"]:
            nodes.append(Node(id=task["name"], label=task["name"], size=15, color="#2196F3"))
            edges.append(Edge(source=task["agent"], target=task["name"], label="Assigné"))
    elif framework == "langgraph":
        for agent in config["agents"]:
            nodes.append(Node(id=agent["name"], label=agent["role"], size=25, color="#4CAF50"))
        for node in config["nodes"]:
            nodes.append(Node(id=node["name"], label=node["name"], size=15, color="#2196F3"))
            edges.append(Edge(source=node["agent"], target=node["name"], label="Gère"))
        for edge in config["edges"]:
            edges.append(Edge(source=edge["source"], target=edge["target"], label=edge.get("condition", "Suivant")))
    config_graph = Config(width=600, height=400, directed=True, physics=True, hierarchical=False)
    return agraph(nodes=nodes, edges=edges, config=config_graph)

def login_page():
    st.title("Connexion pour Ollama")
    with st.form(key="login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")

        if submit:
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.password = password
                st.success("Connexion réussie ! Ollama est maintenant disponible.")
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")

def main_app():
    st.set_page_config(page_title="Advanced Agent Generator", page_icon="🤖", layout="wide")

    st.title("Générateur d'Agents Multi-Framework")
    st.write("Créez des agents IA avec CrewAI, LangGraph ou ReAct. Fournissez votre clé API (connexion requise pour Ollama).")

    # Bouton de déconnexion si connecté
    if st.session_state.get('logged_in', False):
        if st.sidebar.button("Se déconnecter"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.password = ""
            st.success("Déconnexion réussie.")
            st.rerun()

    st.sidebar.title("🔄 Sélection du Framework")
    framework = st.sidebar.radio(
        "Choisissez un framework :", ["crewai", "langgraph", "react"],
        format_func=lambda x: {"crewai": "CrewAI", "langgraph": "LangGraph", "react": "ReAct Framework"}[x]
    )

    st.sidebar.title("🌐 Sélection du Provider")
    public_providers = ["openai", "mistral", "anthropic", "google"]
    provider = st.sidebar.selectbox("Choisissez un provider :", public_providers, index=0)
    api_key = st.sidebar.text_input(f"Clé API pour {provider.capitalize()}", type="password")

    # Option pour Ollama avec connexion requise
    with st.sidebar.expander("Accès avancé"):
        st.write("Utiliser Ollama (connexion requise)")
        if st.session_state.get('logged_in', False) and st.session_state.get('username') == VALID_USERNAME and st.session_state.get('password') == VALID_PASSWORD:
            if st.checkbox("Activer Ollama"):
                provider = "ollama"
                api_key = None
                st.success("Ollama activé !")
        else:
            st.warning("Veuillez vous connecter pour utiliser Ollama.")

    framework_descriptions = {
        "crewai": "**CrewAI** : Orchestre des agents IA autonomes avec des rôles et tâches.",
        "langgraph": "**LangGraph** : Construit des applications multi-acteurs avec état via LLMs.",
        "react": "**ReAct** : Combine raisonnement et action pour des agents LLM."
    }
    st.sidebar.markdown(framework_descriptions[framework])

    st.sidebar.title("📚 Exemples de Prompts")
    try:
        with open("../prompts.json", "r", encoding="utf-8") as f:
            example_prompts = json.load(f)
    except FileNotFoundError:
        st.warning("Fichier '../prompts.json' introuvable. Utilisation des prompts par défaut.")
        example_prompts = {
            "Assistant de Recherche": "J'ai besoin d'un assistant de recherche pour résumer des articles académiques et répondre aux questions.",
            "Créateur de Contenu": "Je veux une équipe pour générer des posts viraux sur les réseaux sociaux et gérer la présence de la marque.",
            "Analyste de Données": "J'ai besoin d'une équipe pour analyser les données clients et créer des visualisations.",
        }
    except json.JSONDecodeError as e:
        st.error(f"Erreur de parsing du fichier JSON : {e}. Utilisation des prompts par défaut.")
        example_prompts = {
            "Assistant de Recherche": "J'ai besoin d'un assistant de recherche pour résumer des articles académiques et répondre aux questions.",
            "Créateur de Contenu": "Je veux une équipe pour générer des posts viraux sur les réseaux sociaux et gérer la présence de la marque.",
            "Analyste de Données": "J'ai besoin d'une équipe pour analyser les données clients et créer des visualisations.",
        }

    selected_example = st.sidebar.selectbox("Choisissez un exemple :", list(example_prompts.keys()))

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎯 Définissez Vos Besoins")
        user_prompt = st.text_area("Décrivez ce dont vous avez besoin :", value=example_prompts[selected_example], height=150)

        if st.button("💡 Suggérer des Améliorations"):
            with st.spinner("Analyse de votre prompt..."):
                generator = AgentGenerator(provider, api_key)
                suggestion = generator.suggest_prompt_improvements(user_prompt)
                st.info(f"**Suggestions :** {suggestion}")

        if st.button(f"🚀 Générer le Code {framework.upper()}"):
            with st.spinner(f"Génération du code {framework.upper()}..."):
                generator = AgentGenerator(provider, api_key)
                config = generator.analyze_prompt(user_prompt, framework)
                try:
                    code = create_code_block(config, framework, provider, api_key)
                    st.session_state.config = config
                    st.session_state.code = code
                    st.session_state.framework = framework
                    st.session_state.provider = provider
                    time.sleep(0.5)
                    st.success(f"✨ Code {framework.upper()} généré avec succès !")
                except Exception as e:
                    st.error(f"Erreur lors de la génération du code : {e}")

    with col2:
        st.subheader("💡 Conseils sur le Framework")
        if framework == "crewai":
            st.info("Définissez des rôles clairs, des objectifs et des règles de collaboration.")
        elif framework == "langgraph":
            st.info("Concevez un flux de graphe logique avec des rôles de nœuds clairs.")
        else:
            st.info("Concentrez-vous sur les étapes de raisonnement et l'intégration des outils.")

    if all(key in st.session_state for key in ['config', 'code', 'framework', 'provider']):
        st.subheader("🔍 Configuration Générée")
        tab1, tab2, tab3 = st.tabs(["📊 Graphe Visuel", "🔧 Détails Config", "💻 Code"])

        with tab1:
            if st.session_state.framework in ["crewai", "langgraph"]:
                st.write("**Graphe de la Structure des Agents**")
                display_graph(st.session_state.config, st.session_state.framework)
            else:
                st.info("Visualisation graphique non disponible pour ReAct.")

        with tab2:
            st.json(st.session_state.config)

        with tab3:
                    st.code(st.session_state.code, language="python")
                    buf = io.StringIO()
                    buf.write(st.session_state.code)
                    st.download_button(
                        label="📥 Télécharger le Code",
                        data=buf.getvalue(),
                        file_name=f"{st.session_state.framework}_agent.py",
                        mime="text/python"
                    )
                    st.info("Téléchargez le code et exécutez-le localement.")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.password = ""

    # Si pas connecté et qu'on veut Ollama, afficher la page de connexion
    if not st.session_state.logged_in and st.session_state.get('show_login', False):
        login_page()
        if st.session_state.logged_in:
            st.session_state.show_login = False  # Réinitialiser après connexion réussie
            st.rerun()
    else:
        main_app()

    # Contrôle pour afficher la page de connexion si Ollama est requis
    if not st.session_state.logged_in and st.sidebar.button("Se connecter pour Ollama"):
        st.session_state.show_login = True
        st.rerun()

if __name__ == "__main__":
    main()