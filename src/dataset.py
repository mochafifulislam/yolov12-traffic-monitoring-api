import os
from roboflow import Roboflow

def download_roboflow_dataset(api_key: str, workspace: str, project_name: str, version_num: int):
    """
    Mengunduh dataset dari Roboflow dan mengembalikan path data.yaml
    """
    print(f"Mengunduh dataset '{project_name}' versi {version_num} dari Roboflow...")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        version = project.version(version_num)
        dataset = version.download("yolov12")
        
        dataset_location = dataset.location
        dataset_yaml_path = os.path.join(dataset_location, "data.yaml")
        
        if os.path.exists(dataset_yaml_path):
            print(f"✅ Dataset berhasil diunduh ke: {dataset_location}")
            print(f"✅ Path data.yaml: {dataset_yaml_path}")
            return dataset_yaml_path, version.version
        else:
            raise FileNotFoundError("data.yaml tidak ditemukan setelah ekstraksi.")
            
    except Exception as e:
        print(f"❌ Gagal mengunduh dataset dari Roboflow: {e}")
        return None, None
