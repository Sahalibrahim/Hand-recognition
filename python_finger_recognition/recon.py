import cv2
import mediapipe as mp

# Initialize camera
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)  # Use 0 for default camera
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)  # Track only one hand
mpDraw = mp.solutions.drawing_utils

# Tip IDs for thumb, index, middle, ring, pinky
tipIds = [4, 8, 12, 16, 20]

def fingers_up(lmList):
    fingers = []
    
    # Thumb: Check if it's open or closed (compare x-coordinates)
    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
        fingers.append(1)  # Thumb is up
    else:
        fingers.append(0)  # Thumb is down

    # Other fingers: Check if they are open (compare y-coordinates)
    for id in range(1, 5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
            fingers.append(1)  # Finger is up
        else:
            fingers.append(0)  # Finger is down

    return fingers

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to find hands
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    if len(lmList) != 0:
        fingers = fingers_up(lmList)
        total_fingers = fingers.count(1)


        cv2.putText(img, f'Fingers: {total_fingers}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 255, 0), 3)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()