import argparse
import cv2
import time
import torch
import os
import sys
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SimpleYOLO, MobileNetYOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='simple', 
                        choices=['simple', 'mobilenet', 'yolo26', 'yolo12', 'rf-detr'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Launching pipeline using: {args.model.upper()}")

    # ==========================================
    # TRACK 1: SOTA SDK Initialization
    # ==========================================
    sota_model = None
    if args.model in ['yolo26', 'yolo12']:
        from ultralytics import YOLO
        weight_name = 'yolo26n.pt' if args.model == 'yolo26' else 'yolov12n.pt'
        sota_model = YOLO(weight_name)

    # ==========================================
    # TRACK 2: Custom PyTorch Initialization
    # ==========================================
    elif args.model == 'mobilenet':
        model = MobileNetYOLO().to(device)
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'room_yolo_weights_mobilenet.pth')
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        model = SimpleYOLO().to(device)
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'room_yolo_weights_simple.pth')
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])

    # ==========================================
    # Camera Initialization
    # ==========================================
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    confidence_threshold = 0.50
    buffer_count = 0
    BUFFER_MAX = 10
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        h_orig, w_orig, _ = frame.shape
        is_occupied = False
        raw_conf = 0.0
        best_box = None

        # ==========================================
        # SOTA INFERENCE TRACK
        # ==========================================
        if args.model in ['yolo26', 'yolo12']:
            results = sota_model(frame, classes=[0], verbose=False)
            if len(results[0].boxes) > 0:
                raw_conf = float(results[0].boxes.conf[0])
                if raw_conf >= confidence_threshold:
                    # YOLO returns [x1, y1, x2, y2] natively
                    best_box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)

        # ==========================================
        # CUSTOM MODEL INFERENCE TRACK
        # ==========================================
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(rgb_frame).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(input_tensor).cpu()[0]

            raw_conf = torch.sigmoid(prediction[0]).item()
            if raw_conf >= confidence_threshold:
                box_coords = prediction[1:5].numpy()
                x_center, y_center, w, h = (int(c * s) for c, s in zip(box_coords, [w_orig, h_orig, w_orig, h_orig]))
                best_box = [max(0, int(x_center - w/2)), max(0, int(y_center - h/2)),
                            min(w_orig, int(x_center + w/2)), min(h_orig, int(y_center + h/2))]

        # ==========================================
        # State Machine & HUD Rendering
        # ==========================================
        if raw_conf >= confidence_threshold:
            buffer_count = min(BUFFER_MAX, buffer_count + 1)
        else:
            buffer_count = max(0, buffer_count - 1)

        is_occupied = (buffer_count >= BUFFER_MAX)
        
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        # HUD Rendering
        cv2.rectangle(frame, (0, 0), (w_orig, 50), (0, 0, 0), -1)
        status_text = f"OCCUPIED ({raw_conf*100:.1f}%)" if is_occupied else f"VACANT ({raw_conf*100:.1f}%)"
        status_color = (0, 255, 0) if is_occupied else (180, 180, 180)
        
        cv2.putText(frame, status_text, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (650, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        if is_occupied:
            cv2.circle(frame, (w_orig - 30, 25), 10, (0, 0, 255), -1)

        if best_box is not None:
            cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 3)

        cv2.imshow("Room Presence Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()