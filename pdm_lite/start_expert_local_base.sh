#!/bin/bash

# This script starts PDM-Lite and the CARLA simulator on a local machine

# Make sure any previously started Carla simulator instance is stopped
# Sometimes calling pkill Carla only once is not enough.
# pkill Carla
# pkill Carla
# pkill Carla

# term() {
#   echo "Terminated Carla"
#   pkill Carla
#   pkill Carla
#   pkill Carla
#   exit 1
# }
# trap term SIGINT

# # carla
# export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export REPETITIONS=1
export DEBUG_CHALLENGE=0

# # Function to handle errors
# handle_error() {
#   pkill Carla
#   exit 1
# }

# # Set up trap to call handle_error on ERR signal
# trap 'handle_error' ERR

# # Start the carla server
# export PORT=$((RANDOM % (40000 - 2000 + 1) + 2000)) # use a random port
# sh ${CARLA_SERVER} -carla-streaming-port=0 -carla-rpc-port=${PORT} &
# sleep 20 # on a fast computer this can be reduced to sth. like 6 seconds

export PORT=2000

echo 'Port' $PORT

export TM_PORT=$((PORT + 3))
export TEAM_AGENT=${WORK_DIR}/team_code/data_agent.py # use autopilot.py here to only run the expert without data generation
# export CHALLENGE_TRACK_CODENAME=SENSORS
export CHALLENGE_TRACK_CODENAME=MAP

export PTH_ROUTE=${WORK_DIR}/leaderboard/data/routes_training
# export PTH_ROUTE=${WORK_DIR}/data
export ROUTES=/home/nupdm/workspace/pdm_lite/leaderboard/data/routes_training.xml

export CHECKPOINT_ENDPOINT=${PTH_ROUTE}.json
export TEAM_CONFIG=${PTH_ROUTE}.xml
export PTH_LOG="/home/nupdm/Datasets/nuPDM/routes_training_new"
export SAVE_PATH="/home/nupdm/Datasets/nuPDM/routes_training_new"
export TM_SEED=0
export DATAGEN=1

# void for false, else for true
export RESUME=

# Start the actual evaluation / data generation
python leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${PORT} --traffic-manager-port=${TM_PORT} --routes=${ROUTES} --repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=0 \
--resume=${RESUME} --timeout=2000 --traffic-manager-seed=${TM_SEED}
# Kill the Carla server afterwards
pkill Carla
