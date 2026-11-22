from pathlib import Path

# Raíz del proyecto 
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos y salidas
DATA_RAW_PATH = BASE_DIR / "data" / "raw" / "bank-full.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PAPER_DIR = BASE_DIR / "paper"

# Semilla fija para reproducibilidad y división de datos
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

# Bandera opcional para probar o discutir el impacto de 'duration'
DROP_DURATION = False

# Asegura que existan las carpetas de salida al importar la configuración
for folder in (DATA_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, PAPER_DIR):
    folder.mkdir(parents=True, exist_ok=True)