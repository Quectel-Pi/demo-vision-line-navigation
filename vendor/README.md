# Model A SDK

`hiwonder/ros_robot_controller_sdk.py` is the existing Hiwonder Model A controller SDK used by this project for serial motor and PWM-servo commands. Its source header is retained. The `line_follow_demo.motor.hiwonder_model_a` adapter loads this file from the project directory.

The standalone servo-test functions and direct-execution motor example at the end of the original SDK have been removed. The protocol definitions and Board implementation are unchanged.

The SDK filename reflects its original controller naming; running this demo does not require ROS. The development-only reference projects are kept separately from the published application.
