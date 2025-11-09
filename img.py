import cv2
import os

# Ask for the name
name = input("Enter your name: ")

# Create a folder with the given name if it doesn't exist
folder_path = os.path.join(os.getcwd(), name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{name}' created.")
else:
    print(f"Folder '{name}' already exists. Images will be added to it.")

# Initialize webcam
cap = cv2.VideoCapture(1)
count = 0
total_images = 30

print("Starting image capture. Press 'q' to quit early.")

while count < total_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    # Display the frame
    cv2.imshow("Capture Images", frame)

    # Save the image
    img_name = f"{name}_{count+1}.jpg"
    cv2.imwrite(os.path.join(folder_path, img_name), frame)
    print(f"Captured image {count+1}/{total_images}")

    count += 1

    # Wait for 100ms and check if 'q' is pressed to quit early
    if cv2.waitKey(100) & 0xFF == ord('q'):
        print("Early exit requested.")
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
print("Image capture completed!")
