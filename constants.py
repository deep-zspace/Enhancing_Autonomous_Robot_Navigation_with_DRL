# Scaling factor to adjust the size and dynamics of the environment
SCALE_FACTOR = 1  # Adjust this value to scale the environment up or down

# Environment dimensions scaled by the scaling factor
ENV_WIDTH = int(600 * SCALE_FACTOR)
ENV_HEIGHT = int(500 * SCALE_FACTOR)

# Robot dimensions scaled by the scaling factor
ROBOT_RADIUS = int(20 * SCALE_FACTOR)
WHEEL_WIDTH = int(5 * SCALE_FACTOR)
WHEEL_HEIGHT = int(10 * SCALE_FACTOR)
AXEL_LENGTH = int(70 * SCALE_FACTOR)

# Robot link lengths scaled by the scaling factor
LINK_LENGTH_MIN = int(50 * SCALE_FACTOR)
LINK_LENGTH_MAX = int(120 * SCALE_FACTOR)

# Maximum wheel velocity - might not need scaling but adjusted for demonstration
MAX_WHEEL_VELOCITY = 10 * SCALE_FACTOR

# Checkpoint radius scaled by the scaling factor
CHECKPOINT_RADIUS = int(25 * SCALE_FACTOR)
