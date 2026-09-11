from ultralytics import YOLO

def initialize_model(model_name: str = "yolo12n.pt"):
    """
    Memuat pretrained model YOLOv12
    """
    try:
        model = YOLO(model_name)
        print(f"✅ Pretrained Model {model_name} berhasil dimuat.")
        return model
    except Exception as e:
        print(f"❌ Gagal memuat model {model_name}: {e}")
        return None
