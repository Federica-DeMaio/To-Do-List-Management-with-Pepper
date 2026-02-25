import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    disable_nnpack = SetEnvironmentVariable(
        name='NNPACK_DISABLE',
        value='1'
    )   

    disable_xnnpack = SetEnvironmentVariable(
        name='XNNPACK_DISABLE',
        value='1'
    )
    
    # --- 1. CONFIGURAZIONE PATH ---
    # Troviamo dove è installato il pacchetto 'pepper_nodes'
    pepper_nodes_share = get_package_share_directory('pepper_nodes')
    
    # Definiamo il percorso al file di lancio ESISTENTE
    bringup_launch_path = os.path.join(pepper_nodes_share, 'launch', 'pepper_bringup.launch.py')

    # Definiamo il percorso di default per il file di config
    default_config_path = os.path.join(pepper_nodes_share, 'conf', 'pepper_params.yaml')

    # --- 2. GESTIONE ARGOMENTI ---
    # Dichiariamo l'argomento anche qui, così se vuoi puoi cambiarlo lanciando questo file
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_path,
        description='Path to the YAML config file with Pepper IP and port'
    )

    # --- 3. INCLUSIONE (La Matrioska) ---
    # Qui diciamo a ROS: "Esegui il launch file di pepper_nodes"
    pepper_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(bringup_launch_path),
        launch_arguments={
            # Passiamo l'argomento 'config_file' dal nostro launch file a quello incluso
            'config_file': LaunchConfiguration('config_file')
        }.items()
    )

    # --- 4. I TUOI NODI ---
    orchestrator_node = Node(
        package='project_nodes',
        executable='orchestrator',
        name='orchestrator',
        output='screen'
    )
    
    # Nodo Audio Pipeline (VAD + SpeechBrain + ASR Logic)
    audio_node = Node(
        package='project_nodes',
        executable='audio_pipeline', # Definito in setup.py
        name='audio_pipeline',
        output='screen'
    )

    # Nodo Facial Recognition (Riconoscimento visivo)
    facial_node = Node(
        package='project_nodes',
        executable='facial_recognition', # Definito in setup.py
        name='facial_recognition',
        output='screen'
    )
    
    # Nodo LLM
    llm_node = Node(
        package='project_nodes',
        executable='llm_node', # Definito in setup.py
        name='llm_node',
        output='screen'
    )
    
    # --- 5. RITORNO DELLE ISTRUZIONI ---
    return LaunchDescription([
        disable_nnpack,
        disable_xnnpack,
        config_file_arg,      # 1. Carica configurazione
        pepper_bringup,       # 2. Avvia Robot Hardware & Base
        orchestrator_node,    # 3. Avvia Orchestrator
        audio_node,           # 4. Avvia Audio Pipeline
        facial_node,          # 5. Avvia Facial Recognition
        llm_node              # 6. Avvia LLM
    ])