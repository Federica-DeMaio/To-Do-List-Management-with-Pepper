import httpx
import os
import json
from pathlib import Path
from project_interfaces.srv import LLMInteraction # type: ignore
import rclpy
from rclpy.node import Node

# --- CONFIGURAZIONE PERCORSI ---
HOME_DIR = str(Path.home())
PROJECT_DATA_DIR = os.path.join(HOME_DIR, '.pepper_project_data')
PATH_TODO_LIST_FILE = os.path.join(PROJECT_DATA_DIR, "todo_list.json")

# Creiamo la cartella se non esiste
os.makedirs(PROJECT_DATA_DIR, exist_ok=True)

# Recupero token per accesso alle api github del modello LLM
# Usiamo get() per evitare crash immediati se la variabile non è esportata,
# anche se le API falliranno successivamente senza token.
token = os.environ.get("GITHUB_TOKEN", "") 

url = "https://models.github.ai/inference/chat/completions"   # link github al model usato per interrogare il modello
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer {}".format(token)
}   # metadati da includere in ogni interrogazione al modello

############### Gestione to-do list ####################
class TodoList():
    """
        Classe per gestire una To-Do List semplice.
        Metodi:
        - _add_user(user: str): Aggiunge un nuovo utente alla lista.
        - get_items() -> list[str] | None: Restituisce gli elementi dell'utente corrente.
        - add_item(item: str): Aggiunge un elemento alla lista dell'utente corrente.
        - remove_item(item_index: int): Rimuove un elemento dalla lista dell'utente corrente in base all'indice.
        - set_current_user(user: str): Imposta l'utente corrente.
        - write_to_file(filename: str): Salva la lista su un file JSON.
        - load_from_file(filename: str) -> 'TodoList': Carica la lista da un file JSON.
    """
    def __init__(self):
        self._items: dict[str, list[str]] = {}
        self._current_user = "unknown"

    def _add_user(self, user: str):
        self._items[user] = []

    def get_items(self) -> list[str] | None:
        return_list = self._items.get(self._current_user, None)
        if not return_list:
            self._add_user(self._current_user)

        return return_list
    
    def add_item(self, item: str):
        if self._current_user in self._items:
            self._items[self._current_user].append(item)
        else:
            raise ValueError(f"Utente {self._current_user} non trovato.")
        
    def remove_item(self, item_index: int) -> str:
        if self._current_user in self._items:
            try:
                return self._items[self._current_user].pop(item_index)
            except IndexError:
                raise ValueError(f"Item index {item_index} non trovato per l'utente {self._current_user}.")
        else:
            raise ValueError(f"Utente {self._current_user} non trovato.")

    def set_current_user(self, user: str):
        if user not in self._items.keys():
            self._add_user(user)
        self._current_user = user
    
    def write_to_file(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self._items, f)

    @staticmethod
    def load_from_file(filename: str) -> 'TodoList':
        todo_list = TodoList()
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                todo_list._items = data
        except FileNotFoundError:
            pass  # Se il file non esiste, ritorna una lista vuota
        return todo_list

TODO_LIST = TodoList.load_from_file(PATH_TODO_LIST_FILE)


############# Funzioni per la gestione della To-Do List #####################

def add_task(item: str) -> str:
    TODO_LIST.add_item(item)
    return f"{item} è stato aggiunto alla lista"

def remove_task(item_index: int) -> str:
    """Rimuove un task dalla lista."""
    return f"🗑️ Task '{TODO_LIST.remove_item(item_index)}' rimosso."

def show_list() -> str:
    user_list = TODO_LIST.get_items()
    if not user_list:  # se la lista è vuota
        return "La lista è vuota"
    list_output = "\n".join([f"{i+1}. {item}" for i, item in enumerate(user_list)])
    return f"📋 La tua To-Do List:\n{list_output}"

def empty_list() -> str:
    TODO_LIST._items[TODO_LIST._current_user] = []
    return "🗑️ La To-Do List è stata svuotata."

#############################################################

def compose(greeting: str) -> str:
    """Restituisce il saluto generato dal modello."""
    return greeting

############## Schema JSON per la formattazione delle risposte fornite dal modello #####################

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "add_task",
            "description": "Aggiunge un nuovo elemento alla To-Do List. Usa questa funzione quando l'utente vuole aggiungere qualcosa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {  # questo campo deve avere il nome uguale al parametro che prendono le funzioni in ingresso
                        "type": "string",
                        "description": "Il nome completo del task da aggiungere (es. 'Comprare il pane al supermercato')."
                    }
                },
                "required": ["item"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_task",
            "description": "Rimuove un elemento specifico dalla To-Do List. Usa questa funzione quando l'utente vuole eliminare o cancellare un task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_index": {
                        "type": "integer",
                        "description": "L'indice del task da rimuovere (es. 0 per il primo task)."
                    }
                },
                "required": ["item_index"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_list",
            "description": "Mostra la To-Do List attuale. Usa questa funzione quando l'utente chiede di vedere, mostrare o leggere la lista.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "empty_list",
            "description": "Svuota completamente la To-Do List. Usa questa funzione quando l'utente vuole cancellare tutti i task.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }, 
    {
        "type": "function",
        "function": {
            "name": "identify_user",
            "description": "Identifica l'utente con cui stai interagendo: devi estrarre il suo nome dalle informazioni fornite, e passarlo alla funzione. Come secondo argomento, devi fornire un saluto personalizzato per l'utente. Usa questa funzione solo quando ti è segnalato da un messaggio del sistema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_name": {
                        "type": "string",
                        "description": "Il nome dell'utente."
                    },
                    "user_salute": {
                        "type": "string",
                        "description": "Un saluto personalizzato per l'utente."
                    }
                },
                "required": ["user_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "greeting_user",
            "description": "Saluta un utente conosciuto con il suo nome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "greeting": {  
                        "type": "string",
                        "description": "Genera un saluto personalizzato con il nome dell'utente."
                    }
                },
                "required": ["greeting"]
            }
        }
    }
] 


# Mapping delle funzioni per l'esecuzione
available_functions = {
    "add_task": add_task,
    "remove_task": remove_task,
    "show_list": show_list,
    "empty_list": empty_list,
    "identify_user": lambda user_name, user_salute: (user_name, user_salute),
    "greeting_user": compose
}

#############################################################################################

######### Classe #########
class LLM(Node):
    def __init__(self):
        super().__init__('llm')
        self.srv = self.create_service(LLMInteraction, 'llm_interaction_service', self.send_interaction)
        self.get_logger().info('Servizio LLMInteraction pronto a ricevere richieste.')
        
    def send_interaction(self, request, response):
        user_input = request.user_input
        user_name = request.user_name # Nome estratto dalla request
        flag = request.start
        user_id = request.user_id

        # Costruiamo il contesto per il System Prompt
        context_info = ""
        
        if user_id == 'unknown':
            context_info = "Non conosci ancora questo utente. Il tuo OBIETTIVO PRIORITARIO è identificarlo. " \
                           "Se l'utente ti dice il suo nome (es. 'Sono Mario', 'Mi chiamo Luca'), " \
                           "DEVI usare il tool 'identify_user'. " \
                           "usa SOLO 'identify_user'."
                           
        else:
            # Carichiamo la lista
            TODO_LIST.set_current_user(user_id)
            current_items = TODO_LIST.get_items() or []
            
            # Istruzione CHIAVE: Diciamo al modello chi è l'utente
            context_info = f"Stai parlando con l'utente registrato di nome '{user_name}'." \
                           f"L'utente può chiedere di aggiornare la sua to do list oppure semplicemente salutarti." \
                           f"Nel primo caso devi gestire la sua To-Do List: [{', '.join(current_items)}] " \
                           f" Se l'utente ti chiede di eseguire più funzioni in una singola richiesta, esegui le funzioni nell'ordine indicato," \
                           f" usando i tool che hai a disposizione." \
                           f" Non devi usare la funzione identify_user perchè il nome dell'utente lo sai già ed è {user_name}." \
                           f" Invece se l'utente ti sta salutando possono essere due casi separati in cui devi comunque usare la funzione 'greeting_user'" \
                           f" I casi si distinguono in base alla variabile flag che ti viene passata." \
                           f" Se flag è true l'utente sta iniziando la conversazione, se è false l'utente la sta concludendo." \
                           f" Nel primo caso, devi salutare l'utente chiamandolo per nome e offrendoti di aiutarlo a gestire la sua to-do list." \
                           f" Nel secondo caso invece lo saluti, augurandogli una buona giornata." \
                           f" Nota che durante le operazioni sulla lista, il flag sarà sempre false, non è rilevante perchè l 'utente non ti sta salutando." \
                           f" Valore del flag da usare solo in caso di saluto: {flag}" \
                           f" Se l'utente ti saluta e richiede di eseguire una operazione sulla lista, salutalo velocemente ed esegui l'operazione, sia in caso di saluto iniziale che finale."

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": f"Sei un androide di nome Pepper. {context_info} " 
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            "tools": tools_schema,
            "temperature": 0.3, # Leggermente più alta per rendere il saluto più naturale
            "top_p": 1.0,
            "max_tokens": 1000,
            "model": "mistral-ai/mistral-medium-2505"
        }

        # =================================================================
        # FASE 1: CONNESSIONE E RETRIEVAL (Logica di Retry)
        # =================================================================
        MAX_RETRIES = 3
        response_data = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self.get_logger().info(f"🔄 Tentativo connessione LLM {attempt}/{MAX_RETRIES}...")
                
                # Timeout di 20 secondi per ogni tentativo
                http_response = httpx.post(url, headers=headers, json=data, timeout=20.0)
                http_response.raise_for_status() # Verifica errori 4xx/5xx
                
                response_data = http_response.json() # Parsing JSON della risposta HTTP
                break # Se siamo qui, successo! Usciamo dal ciclo for
                
            except (httpx.ReadTimeout, httpx.ConnectError, httpx.HTTPStatusError) as e:
                self.get_logger().warn(f"⚠️ Errore di rete (Tentativo {attempt}): {e}")
                if attempt >= MAX_RETRIES:
                    self.get_logger().error("❌ Impossibile contattare il modello LLM dopo vari tentativi.")

        if response_data:
            try:
                # 4.3 Analisi della risposta
                choice = response_data['choices'][0]    # contenuto principale del JSON del contratto API
                
                # Il modello ha deciso di chiamare una funzione
                if choice.get('finish_reason') == 'tool_calls':    
                    message = choice.get('message', {})
                    tool_calls = message.get('tool_calls', [])

                    # Accumulatori per gestire chiamate multiple
                    accumulated_text_responses = []
                    executed_operations = []

                    self.get_logger().info(f"🤖 Il modello ha richiesto {len(tool_calls)} operazioni.")
                    for tool_call in tool_calls:
                        function_name = tool_call['function']['name']
                        try:
                            # La risposta 'args' è spesso una stringa JSON, convertiamola
                            function_args = json.loads(tool_call['function']['arguments'])
                        except json.JSONDecodeError:
                            self.get_logger().error(f"🤖 Errore di parsing: il modello non ha generato JSON valido per {function_name}.")
                            raise ValueError("Errore di Parsing...")

                        # 4.4 Esecuzione della funzione Python
                        if function_name in available_functions:
                            function_to_call = available_functions[function_name] # sfruttiamo mapping per ottenere nome funzione

                            # Eseguiamo la funzione con gli argomenti forniti dall'LLM
                            result_message = function_to_call(**function_args)
                            
                            # Gestione specifica per identify_user (che ritorna una tupla)
                            if function_name == "identify_user":
                                response.item_content = [result_message[0]]  # Nome utente nel content
                                accumulated_text_responses.append(result_message[1])  # Saluto nel testo
                                response.operation = 'identification'
                                executed_operations.append('identification')

                            else:
                                # Funzioni standard (ritornano stringa)
                                accumulated_text_responses.append(result_message)
                                executed_operations.append(function_name)
                            self.get_logger().info(f"🤖 {result_message}")
                        else:
                            raise ValueError(f"🤖 Errore interno: funzione '{function_name}' non definita.")

                    # Costruzione della risposta finale per l'Orchestrator
                    response.text_response = " ".join(accumulated_text_responses)

                    if "identification" in executed_operations:
                        pass  # item_content è già stato settato col nome
                    else:
                        # In tutti gli altri casi, ritorniamo la lista aggiornata
                        response.item_content = TODO_LIST.get_items() or []

                    # Definizione del campo operation
                    if len(executed_operations) > 1:
                        if "greeting_user" in executed_operations:
                            if flag:
                                response.operation = 'multiple_ops'
                            else:
                                response.operation = 'multiple_ops_exit'
                        else:
                            response.operation = 'multiple_ops'

                    elif len(executed_operations) == 1:
                        response.operation = executed_operations[0]
                    else:
                        response.operation = 'none'

                # Il modello ha deciso di rispondere con testo (es. se la richiesta è vaga o se non deve usare un tool)
                else:
                    final_response = choice['message']['content']
                    self.get_logger().info(f"🤖 {final_response}")
                    response.operation = 'repeat'
                    response.text_response = final_response
                    response.item_content = []

            except httpx.HTTPStatusError as e:
                self.get_logger().error(f"\n❌ Errore HTTP: {e.response.status_code} - Controlla il tuo token GITHUB_TOKEN o il nome del modello. {e.response.text}")
                response.operation = 'error'
            except Exception as e:
                self.get_logger().error(f"\n❌ Si è verificato un errore inatteso: {e}")
                response.operation = 'error'
            
            # Salvataggio della To-Do List su file dopo ogni operazione
            TODO_LIST.write_to_file(PATH_TODO_LIST_FILE)
            
        return response

def main():
    rclpy.init()
    llm = LLM()
    rclpy.spin(llm)
    rclpy.shutdown()

if __name__ == '__main__':
    main()