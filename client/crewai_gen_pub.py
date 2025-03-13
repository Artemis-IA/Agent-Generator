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

# Chargement des variables d'environnement
load_dotenv()

# Récupération des identifiants depuis .env (sans valeurs par défaut)
VALID_USERNAME = os.getenv("STREAMLIT_USERNAME")
VALID_PASSWORD = os.getenv("STREAMLIT_PASSWORD")

# Vérification que les credentials sont bien définis
if not VALID_USERNAME or not VALID_PASSWORD:
    raise ValueError("Les variables STREAMLIT_USERNAME et STREAMLIT_PASSWORD doivent être définies dans le fichier .env.")

class AgentGenerator:
    def __init__(self):
        self.model_id = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.parameters = {
            "temperature": 0.7,
            "max_tokens": 1500,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        self.model = None

    def _initialize_model(self):
        if self.model is None:
            try:
                ollama.list()
                self.model = ollama
            except Exception as e:
                st.error(f"Erreur : Ollama n'est pas détecté. Assurez-vous qu'il est installé et en cours d'exécution. Détails : {e}")
                st.stop()

    def analyze_prompt(self, user_prompt: str, framework: str) -> Dict[str, Any]:
        self._initialize_model()
        system_prompt = self._get_system_prompt_for_framework(framework)
        try:
            formatted_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.model.generate(
                model=self.model_id,
                prompt=formatted_prompt,
                options=self.parameters
            )["response"]
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
        response = self.model.generate(
            model=self.model_id,
            prompt=suggestion_prompt,
            options={"max_tokens": 200}
        )["response"]
        return response

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

def create_code_block(config: Dict[str, Any], framework: str) -> str:
    if framework == "crewai":
        return create_crewai_code(config)
    elif framework == "langgraph":
        return create_langgraph_code(config)
    elif framework == "react":
        return create_react_code(config)
    return "# Framework invalide"

def create_crewai_code(config: Dict[str, Any]) -> str:
    code = """from crewai import Agent, Task, Crew
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

"""
    for agent in config["agents"]:
        role = repr(agent["role"])
        goal = repr(agent["goal"])
        backstory = repr(agent["backstory"])
        code += f"agent_{agent['name']} = Agent(\n    role={role},\n    goal={goal},\n    backstory={backstory},\n    verbose={agent['verbose']},\n    allow_delegation={agent['allow_delegation']},\n    tools=[search_tool, wiki_tool]\n)\n\n"
    
    for task in config["tasks"]:
        task_name = re.sub(r'\s+', '_', task["name"])
        description = repr(task["description"])
        expected_output = repr(task["expected_output"])
        code += f"task_{task_name} = Task(\n    description={description},\n    agent=agent_{task['agent']},\n    expected_output={expected_output}\n)\n\n"
    
    code += "crew = Crew(\n    agents=[" + ", ".join(f"agent_{a['name']}" for a in config["agents"]) + "],\n    tasks=[" + ", ".join(f"task_{re.sub(r'\s+', '_', t['name'])}" for t in config["tasks"]) + "]\n)\nresult = crew.kickoff()\n\nprint(result)"
    return code

def create_langgraph_code(config: Dict[str, Any]) -> str:
    code = """from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from typing import Dict, List, Any, TypedDict

class AgentState(TypedDict):
    messages: List[BaseMessage]
    next: str

search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [search_tool, wiki_tool]

"""
    for agent in config["agents"]:
        llm_model = agent.get('llm', 'llama3.2')
        code += f"def {agent['name']}_agent(state: AgentState) -> AgentState:\n    llm = ChatOpenAI(model=\"{llm_model}\")\n    messages = state['messages']\n    response = llm.invoke(messages)\n    return {{\"messages\": messages + [response], \"next\": state.get(\"next\", \"\")}}\n\n"
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

def create_react_code(config: Dict[str, Any]) -> str:
    code = """from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

search_tool = DuckDuckGoSearchRun()
wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [search_tool, wiki_tool]

"""
    if config.get("agents"):
        agent = config["agents"][0]
        llm_model = agent.get('llm', 'llama3.2')
        role = repr(agent["role"])
        goal = repr(agent["goal"])
        code += f"""llm = ChatOpenAI(model="{llm_model}")

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
    return code

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
    st.title("Connexion")
    with st.form(key="login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submit = st.form_submit_button("Se connecter")

        if submit:
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.logged_in = True
                st.success("Connexion réussie !")
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")

def main_app():
    st.set_page_config(page_title="Advanced Agent Generator", page_icon="🤖", layout="wide")

    st.title("Générateur d'Agents Multi-Framework")
    st.write("Créez des agents IA avec CrewAI, LangGraph ou ReAct. Téléchargez le code pour l'exécuter localement.")

    st.sidebar.title("🔄 Sélection du Framework")
    framework = st.sidebar.radio(
        "Choisissez un framework :", ["crewai", "langgraph", "react"],
        format_func=lambda x: {"crewai": "CrewAI", "langgraph": "LangGraph", "react": "ReAct Framework"}[x]
    )

    framework_descriptions = {
        "crewai": "**CrewAI** : Orchestre des agents IA autonomes avec des rôles et tâches.",
        "langgraph": "**LangGraph** : Construit des applications multi-acteurs avec état via LLMs.",
        "react": "**ReAct** : Combine raisonnement et action pour des agents LLM."
    }
    st.sidebar.markdown(framework_descriptions[framework])

    st.sidebar.title("📚 Exemples de Prompts")
    example_prompts = {
        "Assistant de Recherche": "J'ai besoin d'un assistant de recherche pour résumer des articles académiques et répondre aux questions.",
        "Créateur de Contenu": "Je veux une équipe pour générer des posts viraux sur les réseaux sociaux et gérer la présence de la marque.",
        "Analyste de Données": "J'ai besoin d'une équipe pour analyser les données clients et créer des visualisations.",
        "Rédacteur Technique": "J'ai besoin d'une équipe pour rédiger des documentations techniques et des guides API.",
        "Support Client": "Créez une équipe de support pour gérer les requêtes clients et escalader les problèmes.",
    }
    selected_example = st.sidebar.selectbox("Choisissez un exemple :", list(example_prompts.keys()))

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎯 Définissez Vos Besoins")
        user_prompt = st.text_area("Décrivez ce dont vous avez besoin :", value=example_prompts[selected_example], height=150)

        if st.button("💡 Suggérer des Améliorations"):
            with st.spinner("Analyse de votre prompt..."):
                generator = AgentGenerator()
                suggestion = generator.suggest_prompt_improvements(user_prompt)
                st.info(f"**Suggestions :** {suggestion}")

        if st.button(f"🚀 Générer le Code {framework.upper()}"):
            with st.spinner(f"Génération du code {framework.upper()}..."):
                generator = AgentGenerator()
                config = generator.analyze_prompt(user_prompt, framework)
                st.session_state.config = config
                st.session_state.code = create_code_block(config, framework)
                st.session_state.framework = framework
                time.sleep(0.5)
                st.success(f"✨ Code {framework.upper()} généré avec succès !")

    with col2:
        st.subheader("💡 Conseils sur le Framework")
        if framework == "crewai":
            st.info("Définissez des rôles clairs, des objectifs et des règles de collaboration.")
        elif framework == "langgraph":
            st.info("Concevez un flux de graphe logique avec des rôles de nœuds clairs.")
        else:
            st.info("Concentrez-vous sur les étapes de raisonnement et l'intégration des outils.")

    if 'config' in st.session_state:
        st.subheader("🔍 Configuration Générée")
        tab1, tab2, tab3 = st.tabs(["📊 Graphe Visuel", "🔧 Détails Config", "💻 Code"])

        with tab1:
            if framework in ["crewai", "langgraph"]:
                st.write("**Graphe de la Structure des Agents**")
                display_graph(st.session_state.config, st.session_state.framework)
            else:
                st.info("Visualisation graphique non disponible pour ReAct.")

        with tab2:
            st.json(st.session_state.config)

        with tab3:
            edited_code = st.text_area("Éditez le code généré ici :", value=st.session_state.code, height=400, key="code_editor")
            st.session_state.code = edited_code
            buf = io.StringIO()
            buf.write(st.session_state.code)
            st.download_button(
                label="📥 Télécharger le Code",
                data=buf.getvalue(),
                file_name=f"{framework}_agent.py",
                mime="text/python"
            )
            st.info("Téléchargez le code et exécutez-le localement pour tester votre agent.")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()