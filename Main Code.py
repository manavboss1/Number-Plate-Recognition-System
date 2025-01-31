import cv2
import pytesseract
import requests

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained number plate detection model
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Function to detect and recognize number plates
def recognize_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        plate = frame[y:y+h, x:x+w]
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        plate_text = pytesseract.image_to_string(plate_gray, config='--psm 8').strip()
        
        # Draw a rectangle around the plate
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # Display the recognized text
        cv2.putText(frame, plate_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Fetch driver information from the government site
        driver_info = get_driver_info(plate_text)
        if driver_info:
            cv2.putText(frame, driver_info, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

# Function to get driver information from the government site
def get_driver_info(plate_number):
    # Replace with the actual API endpoint and parameters
    api_url = "https://government-site/api/driver-info"
    params = {"plate_number": plate_number}
    response = requests.get(api_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return f"Name: {data['name']}, Address: {data['address']}, Registration: {data['registration']}"
    else:
        return None

# Main loop to process the video stream
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_plates = recognize_plate(frame)
    cv2.imshow('Number Plate Recognition', frame_with_plates)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
