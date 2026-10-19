import os
import shutil
import random

def move_test_split(frames_dir='dataset_frames', 
                    labels_dir='dataset_labels', 
                    test_frames_dir='test_frames', 
                    test_labels_dir='test_labels', 
                    ratio=0.15, 
                    seed=42):
    
    # Set seed for reproducible random selection
    random.seed(seed)
    
    # Create target directories
    os.makedirs(test_frames_dir, exist_ok=True)
    os.makedirs(test_labels_dir, exist_ok=True)
    
    # Get list of valid images
    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_images = [f for f in os.listdir(frames_dir) if f.lower().endswith(valid_extensions)]
    
    if not all_images:
        print(f"No images found in '{frames_dir}'. Please check the directory path.")
        return

    # Calculate 15% count
    num_to_move = int(len(all_images) * ratio)
    selected_images = random.sample(all_images, num_to_move)
    
    moved_images_count = 0
    moved_labels_count = 0
    
    for img_name in selected_images:
        # Determine image paths
        src_img = os.path.join(frames_dir, img_name)
        dst_img = os.path.join(test_frames_dir, img_name)
        
        # Determine matching label file name
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.txt"
        
        # Explicit safety check: NEVER touch classes.txt
        if label_name.lower() == 'classes.txt':
            continue
            
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(test_labels_dir, label_name)
        
        # Move the image
        shutil.move(src_img, dst_img)
        moved_images_count += 1
        
        # Move the matching label file if it exists
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
            moved_labels_count += 1

    print("--- DATASET SPLIT COMPLETE ---")
    print(f"Total dataset size: {len(all_images)} frames")
    print(f"Moved {moved_images_count} images to '{test_frames_dir}' ({ratio * 100:.0f}%)")
    print(f"Moved {moved_labels_count} label files to '{test_labels_dir}'")
    print(f"Verified: 'classes.txt' remains untouched in '{labels_dir}'")

if __name__ == "__main__":
    move_test_split()