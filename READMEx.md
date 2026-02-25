# Pepper LLM To-Do List Manager 🤖📋

This ROS 2 package (`project_nodes`) manages an intelligent, multi-user To-Do List system using the humanoid robot Pepper. Developed as part of the Cognitive Robotics course, it seamlessly integrates Facial Recognition, advanced Audio Processing, and Large Language Model (LLM) capabilities to provide a natural, context-aware human-robot interaction.

## 🎯 Project Overview

The goal of this project is to empower Pepper to act as a smart personal assistant. By combining visual and auditory perception, the robot can recognize different users, maintain independent memory for each, and update their personal To-Do Lists dynamically on its tablet, all through natural language conversations.

## 🏗️ System Architecture

The architecture relies on the base `pepper_nodes` for hardware interfacing and builds a sophisticated intelligence layer divided into specialized Work Packages (WPs):

* **Orchestrator Node:** The core of the system. It implements a robust **Finite State Machine (FSM)** to manage registration flows, memory allocation, and state transitions using advanced ROS 2 communication paradigms.
* **Vision Module (Facial Recognition):** Identifies users in real-time. It handles dynamic session changes:
  * *New Person Entry:* Automatically clears the tablet UI to protect privacy if an unregistered person steps in.
  * *ID Change:* Instantly loads and displays the correct To-Do List when a different registered user is recognized.
* **Audio Module:** Features Voice Activity Detection (VAD) and Automatic Speech Recognition (ASR). It includes a crucial **Self-Voice Filter**, preventing Pepper from erroneously interpreting its own text-to-speech output as a user command.
* **Dialog & LLM Node:** Processes textual prompts using a Large Language Model. It relies on a strictly defined `tools_schema` to accurately map natural language to specific list operations (**Add**, **Remove**, **View**, and **Empty**). 
* **Awareness Node:** Manages visual tracking, ensuring Pepper fluidly maintains eye contact ("SemiEngaged" mode) with the active speaker.

## ✨ Key Validated Features

Extensive operational testing has validated several advanced capabilities:
* **Interactive Error Handling (Prompt Engineering):** When faced with ambiguous or unrecognized voice commands, Pepper actively asks the user to repeat or clarify the request, ensuring high reliability.
* **Uninterrupted Interaction:** Thanks to the speaker verification and self-voice filtering, users can reply immediately after Pepper finishes speaking without audio interference.
* **Dynamic Tablet UI:** Real-time synchronization between the robot's memory and the tablet display based on the person currently recognized by the Vision Module.

---

## ⚙️ Prerequisites and Installation

### 1. System Requirements
* **ROS 2** (Humble, Iron, or Jazzy)
* **Python** 3.10+
* Network connection to the Pepper robot (or a simulated environment).

### 2. Python Dependencies
This project requires several machine learning, computer vision, and audio processing libraries. Install the dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
