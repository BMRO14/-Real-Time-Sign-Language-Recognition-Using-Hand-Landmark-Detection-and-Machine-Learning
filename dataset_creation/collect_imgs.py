import os
import cv2

DATA_DIR = 'C:/Users/parav/Desktop/NN_Project/data'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
number_of_classes = 26
dataset_size = 200

cap = None
print("Testing cameras...")

for index in [0, 1, 2, 3]:
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        test_cap = cv2.VideoCapture(index, backend)
        if test_cap.isOpened():
            test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None and test_frame.size > 0:
                print(f"✓ Found working camera: index={index}, backend={backend}")
                cap = test_cap
                break
        test_cap.release()
    if cap:
        break

if cap is None or not cap.isOpened():
    print("ERROR: No working camera found!")
    print("1. Check Windows Camera app works")
    print("2. Enable camera in Privacy settings")
    print("3. Close other apps using camera")
    print("4. Update camera drivers")
    exit()

print(f"Using camera index {index}")
print("Camera ready. Press Q when ready to collect for each class.")

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
        
        print(f'\n=== Collecting data for class {j} ===')
        
        print("Press 'Q' when ready to start collecting images...")
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Capture failed")
                continue
                
            cv2.putText(frame, 'When ready press "Q"', (100, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f'Class {j}', (100, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Camera - Press Q to start collecting', frame)
            
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow('Camera - Press Q to start collecting')
                break
            elif key == ord('x'): 
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f"Collecting {dataset_size} images for class {j}...")
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to grab frame")
                continue
                
            cv2.putText(frame, f'Class {j}: {counter}/{dataset_size}', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Collecting Images - Press X to stop', frame)
            
            cv2.imwrite(os.path.join(class_dir, f'{counter:04d}.jpg'), frame)
            counter += 1
            
            key = cv2.waitKey(50) & 0xFF
            if key == ord('x'):  
                break
                
        print(f'✓ Saved {counter} images to {class_dir}')
        
        cv2.waitKey(1000)

cap.release()
cv2.destroyAllWindows()
print(f"\n🎉 Dataset collection complete!")
print(f"📁 Images saved to: {DATA_DIR}")
print(f"📊 Total classes: {number_of_classes}, Images per class: {dataset_size}")
