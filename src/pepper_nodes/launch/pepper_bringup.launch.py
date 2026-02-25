import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = "pepper_nodes"
    config_file = LaunchConfiguration('config_file')
    default_config_path = os.path.join(get_package_share_directory(pkg_name), "conf", "pepper_params.yaml")
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_path,
            description='Path to the YAML config file with Pepper IP and port'
        ),
        Node(
            package=pkg_name,
            executable='wakeup_node',
            name='wakeup_node',
            parameters=[config_file],
            output="screen",
        ),
        Node(
            package=pkg_name,
            executable='text2speech_node',
            name='text2speech_node',
            parameters=[config_file],
            output="screen",
        ),
        Node(
            package=pkg_name,
            executable='tablet_node',
            name='tablet_node',
            parameters=[config_file],
            output="screen",
        ),
        Node(
            package=pkg_name,
            executable='image_input_node',
            name='image_input_node',
            parameters=[config_file],
            output="screen",
        ),
        Node(
            package=pkg_name,
            executable='head_motion_node',
            name='head_motion_node',
            parameters=[config_file],
            output="screen",
        ),
        Node(
            package=pkg_name,
            executable='respeaker_mic',
            name='respeaker_mic',
            parameters=[config_file],
            output="screen",
        ),
        Node(
            package=pkg_name,
            executable='asr_whisper',
            name='asr_whisper',
            parameters=[config_file],
            output="screen",
        ),
        Node(
            package=pkg_name,
            executable='awareness_node',
            name='awareness_node',
            parameters=[config_file],
            output="screen",
        ),
    ])