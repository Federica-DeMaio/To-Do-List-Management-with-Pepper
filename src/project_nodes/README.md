\# Pepper LLM To-Do List Manager 



Questo pacchetto ROS 2 (`project\_nodes`) gestisce l'interazione intelligente con il robot umanoide Pepper. Combina il riconoscimento facciale, l'elaborazione audio avanzata e l'integrazione con un LLM (Large Language Model) per permettere a Pepper di riconoscere gli utenti, interagire in modo naturale e gestire una To-Do List.



\## Architettura del Progetto



Il sistema si appoggia al pacchetto base `pepper\_nodes` per l'interfaccia hardware e aggiunge un livello di intelligenza superiore tramite i seguenti nodi principali:



\* \*\*Orchestrator:\*\* Il cervello centrale che coordina le transizioni di stato e gestisce la logica del sistema.

\* \*\*Audio Pipeline:\*\* Gestisce il VAD (Voice Activity Detection), l'ASR (Automatic Speech Recognition) con SpeechBrain e la logica di trascrizione.

\* \*\*Facial Recognition:\*\* Utilizza la telecamera di Pepper e librerie di computer vision (es. DeepFace) per identificare chi sta parlando.

\* \*\*LLM Node:\*\* Invia i prompt testuali a un modello linguistico per generare risposte naturali e gestire le azioni della To-Do List.

\* \*\*Awareness Node\*\* \*(da `pepper\_nodes`)\*: Gestisce il tracking visivo per far sì che Pepper mantenga il contatto visivo ("SemiEngaged") con l'utente in modo fluido.



---



\## Prerequisiti e Installazione



\### 1. Requisiti di Sistema

\* \*\*ROS 2\*\* (Humble, Iron o Jazzy)

\* Python 3.10+

\* Connessione di rete con il robot Pepper (o un ambiente simulato).



\### 2. Dipendenze Python

Questo progetto richiede diverse librerie di machine learning. Installa le dipendenze usando il file `requirements.txt`:



```bash

pip install -r requirements.txt

