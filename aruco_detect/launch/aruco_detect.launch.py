from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument('camera', default_value='/camera/camera/color', description='Namespace for camera input'),  # edit 
        DeclareLaunchArgument('image', default_value='image_raw', description='Camera topic name'), # edit 
        DeclareLaunchArgument('transport', default_value='raw', description='Image transport method'),   # edit 
        DeclareLaunchArgument('fiducial_len', default_value='0.14', description='Fiducial length in meters'),
        DeclareLaunchArgument('dictionary', default_value='3', description='ArUco dictionary ID'),     # edit
        DeclareLaunchArgument('do_pose_estimation', default_value='true', description='Enable pose estimation'),
        DeclareLaunchArgument('vis_msgs', default_value='false', description='Publish vision_msgs for pose estimation'),
        DeclareLaunchArgument('ignore_fiducials', default_value='', description='Ignore specific fiducials'),
        DeclareLaunchArgument('fiducial_len_override', default_value='', description='Override fiducial length'),
        DeclareLaunchArgument('verbose', default_value='true', description='Enable verbose output'),

        # Define the node
        Node(
            package='aruco_detect',
            executable='aruco_detect',
            name='aruco_detect',
            output='screen',
            parameters=[
                {'image_transport': LaunchConfiguration('transport')},
                {'publish_images': True},
                {'fiducial_len': LaunchConfiguration('fiducial_len')},
                {'dictionary': LaunchConfiguration('dictionary')},
                {'do_pose_estimation': LaunchConfiguration('do_pose_estimation')},
                {'vis_msgs': LaunchConfiguration('vis_msgs')},
                {'ignore_fiducials': LaunchConfiguration('ignore_fiducials')},
                {'fiducial_len_override': LaunchConfiguration('fiducial_len_override')},
                {'verbose': LaunchConfiguration('verbose')},
            ],
            remappings=[
                ('camera/image_raw', [LaunchConfiguration('camera'), '/', LaunchConfiguration('image')]),    # edit 
                ('camera_info', [LaunchConfiguration('camera'), '/camera_info']),   
            ],
            arguments=['--ros-args', '--log-level', 'warn']
        ),
    ])
