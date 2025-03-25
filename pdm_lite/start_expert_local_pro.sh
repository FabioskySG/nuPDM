#!/bin/bash
# This script starts PDM-Lite and the CARLA simulator on a local machine

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export REPETITIONS=1
export DEBUG_CHALLENGE=0
export PTH_ROUTE=${WORK_DIR}/data
export PORT=2000
export TEAM_AGENT=${WORK_DIR}/team_code/data_agent.py
export CHALLENGE_TRACK_CODENAME=MAP
export TM_PORT=$((PORT + 3))
export CHECKPOINT_ENDPOINT=${PTH_ROUTE}.json
export TEAM_CONFIG=${PTH_ROUTE}.xml
export TM_SEED=0
export DATAGEN=1 # void for false, else for true
export RESUME=
export DATA_DIR="/home/nupdm/workspace/pdm_lite/data/50x38_town_12"

# Función para mostrar ayuda
show_help() {
  echo "Uso: $0 [opciones]"
  echo "Opciones:"
  echo "  -h, --help            Muestra esta ayuda"
  echo "  -t, --type TIPO       Especifica un tipo de ruta concreto a procesar"
  echo "  -l, --list            Lista todos los tipos de rutas disponibles"
  echo "Sin opciones, se procesarán todos los tipos de rutas."
}

# Función para listar tipos de rutas disponibles
list_route_types() {
  echo "Tipos de rutas disponibles:"
  for dir in ${DATA_DIR}/*/ ; do
    echo "  - $(basename "${dir}")"
  done
}

# Función para manejar la limpieza al terminar
cleanup() {
  echo "Terminando Carla"
  pkill Carla
  exit 1
}

# Función para procesar un tipo de ruta
process_route_type() {
  local ROUTE_TYPE=$1
  local ROUTE_TYPE_DIR="${DATA_DIR}/${ROUTE_TYPE}/"
  
  if [ ! -d "${ROUTE_TYPE_DIR}" ]; then
    echo "Error: El tipo de ruta '${ROUTE_TYPE}' no existe."
    list_route_types
    exit 1
  fi
  
  echo "Procesando tipo de ruta: ${ROUTE_TYPE}"
  
  # Configurar directorios de salida específicos para este tipo de ruta
  export PTH_LOG="/home/nupdm/Datasets/nuPDM/50x38_town_12/test/${ROUTE_TYPE}"
  export SAVE_PATH="/home/nupdm/Datasets/nuPDM/50x38_town_12/test/${ROUTE_TYPE}"
  
  # Crear directorios si no existen
  mkdir -p "${PTH_LOG}"
  mkdir -p "${SAVE_PATH}"
  
  # Iterar sobre cada archivo en el directorio del tipo de ruta
  for ROUTE_FILE in ${ROUTE_TYPE_DIR}* ; do
    # Ignorar archivos PNG
    if [[ "${ROUTE_FILE}" == *.png ]]; then
      echo "Ignorando archivo PNG: ${ROUTE_FILE}"
      continue
    fi
    
    # Verificar que sea un archivo XML
    if [[ "${ROUTE_FILE}" == *.xml ]] && [ -f "${ROUTE_FILE}" ]; then
      echo "Procesando archivo de ruta: ${ROUTE_FILE}"
      
      # Configurar la ruta actual
      export ROUTES="${ROUTE_FILE}"
      
      # Iniciar el servidor Carla si no está en ejecución
      # if ! pgrep -x "Carla" > /dev/null; then
      #   echo "Iniciando servidor Carla en puerto ${PORT}"
      #   sh ${CARLA_SERVER} -carla-streaming-port=0 -carla-rpc-port=${PORT} &
      #   sleep 20  # Esperar a que Carla se inicie completamente
      # fi
      
      echo "Ejecutando evaluación para ${ROUTE_FILE}"
      # Iniciar la evaluación para esta ruta
      python leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${PORT} --traffic-manager-port=${TM_PORT} \
        --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} \
        --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} \
        --debug=0 --resume=${RESUME} --timeout=2000 --traffic-manager-seed=${TM_SEED}
      
      # Pausa breve entre ejecuciones
      sleep 5
    else
      echo "Ignorando archivo no XML: ${ROUTE_FILE}"
    fi
  done
  
  # Reiniciar Carla después de procesar todas las rutas del tipo
  pkill Carla
  sleep 10
}

# Configurar trap para capturar señales de terminación
trap cleanup SIGINT SIGTERM ERR

# Parsear argumentos de línea de comandos
SPECIFIC_TYPE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    -l|--list)
      list_route_types
      exit 0
      ;;
    -t|--type)
      SPECIFIC_TYPE="$2"
      shift 2
      ;;
    *)
      echo "Opción desconocida: $1"
      show_help
      exit 1
      ;;
  esac
done

# Procesar rutas según la configuración
if [ -n "${SPECIFIC_TYPE}" ]; then
  # Procesar solo el tipo específico
  process_route_type "${SPECIFIC_TYPE}"
else
  # Procesar todos los tipos de rutas
  echo "Procesando todos los tipos de rutas disponibles"
  for ROUTE_TYPE_DIR in ${DATA_DIR}/*/ ; do
    ROUTE_TYPE=$(basename "${ROUTE_TYPE_DIR}")
    process_route_type "${ROUTE_TYPE}"
  done
fi

# Asegurarse de que Carla se cierre al finalizar
pkill Carla
echo "Procesamiento de rutas completado"