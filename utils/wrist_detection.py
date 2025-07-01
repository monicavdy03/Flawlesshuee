import cv2
from ultralytics import YOLO


model = YOLO("model/wrist_best_model.pt")

if (model):
    print("Model is ready")
else:
    print("Model is not loaded")

def predict_wrist(image_path, conf_threshold=0.25):
    """
    Melakukan prediksi pada satu gambar, menampilkan hanya bounding box terbesar (berdasarkan area),
    mencetak koordinat serta confidence-nya, dan memotong gambar hanya menyisakan area dalam bounding box.

    Returns:
        cropped_img (np.ndarray or None): Gambar yang telah dipotong (jika ada deteksi), else None.
    """
    # Lakukan prediksi
    try:
        results = model.predict(source=image_path, conf=conf_threshold, save=False, verbose=False)

        # Ambil gambar asli
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_img = None

        for result in results:
            boxes = result.boxes
            if boxes is None:
                return None

            # Temukan box terbesar
            largest_box = None
            max_area = 0

            for box in boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = coords
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_box = box

            if largest_box:
                coords = largest_box.xyxy[0].cpu().numpy()
                conf = largest_box.conf[0].cpu().numpy()
                cls_id = int(largest_box.cls[0].cpu().numpy())
                x1, y1, x2, y2 = map(int, coords)

                # Crop gambar asli
                cropped_img = img_rgb[y1:y2, x1:x2] 
                return_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)

                # Print informasi deteksi
                print(f"[{model.names[cls_id]}] Box: ({x1}, {y1}, {x2}, {y2}), Confidence: {conf:.2f}")

        return return_img
    
    except:
        return None

