from lunar_workflow import lunar_workflow
from mars_workflow import mars_workflow
import os

# Paths for lunar data
lunar_training_data_path = r'D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar\training\data\S12_GradeA'
lunar_catalog_path = r'D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar\training\catalogs\apollo12_catalog_GradeA_final.csv'
lunar_test_paths = [
    r'D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar\test\data\S12_GradeB',
    r'D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar\test\data\S15_GradeA'
    r'D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar\test\data\S15_GradeB'
    r'D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar\test\data\S16_GradeA'
    r'D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar\test\data\S16_GradeA'

]
lunar_output_path = r'results\lunar'

# Paths for Mars data
mars_training_data_path = r'D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\mars\training\data'
mars_catalog_path = r'D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\mars\training\catalogs\Mars_InSight_training_catalog_final.csv'
mars_test_data_path = r'D:\Nasa space apps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\mars\test\data'
mars_output_path = r'results\mars'

# Ensure output directories exist
os.makedirs(lunar_output_path, exist_ok=True)
os.makedirs(mars_output_path, exist_ok=True)

# Run the lunar workflow
print("Starting Lunar Workflow...")
lunar_workflow(lunar_training_data_path, lunar_catalog_path, lunar_test_paths, lunar_output_path)

# Run the Mars workflow
print("Starting Mars Workflow...")
mars_workflow(mars_training_data_path, mars_catalog_path, mars_test_data_path, mars_output_path)
