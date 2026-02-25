from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'pepper_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
         # Install conf files
        (os.path.join('share', package_name, 'conf'), glob('conf/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mivia',
    maintainer_email='44608428+gdesimone97@users.noreply.github.com',
    description='Nodi ROS 2 di base per l\'interfaccia hardware di Pepper',
    license='Apache-2.0',
    tests_require=['pytest'],    
    entry_points={
        'console_scripts': [
            'wakeup_node = pepper_nodes.wakeup_node:main',
            'text2speech_node = pepper_nodes.text2speech_node:main',
            'tablet_node = pepper_nodes.tablet_node:main',
            'image_input_node = pepper_nodes.image_input_node:main',
            'camera_show_node = pepper_nodes.camera_show_node:main',
            'head_motion_node = pepper_nodes.head_motion_node:main',
            'asr_whisper = pepper_nodes.asr:main',
            'respeaker_mic = pepper_nodes.respeaker.respeaker_node:main',
            'awareness_node = pepper_nodes.awareness_node:main',
        ],
    },
)