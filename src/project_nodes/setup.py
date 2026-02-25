from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'project_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # --- AGGIUNTA FONDAMENTALE PER I LAUNCH FILE ---
        # Dice a ROS2: "Prendi tutti i file in 'launch/' e copiali nella cartella di installazione"
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mivia',
    maintainer_email='mivia@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'orchestrator = project_nodes.orchestrator_node:main',
        'llm_node = project_nodes.llm_node:main',
        'terminal_input = project_nodes.terminal_input_node:main',
        'facial_recognition = project_nodes.facial_recognition:main',
        'audio_pipeline = project_nodes.audio_pipeline_node:main',
        ],
    },
)
