import cv2
class Idk():
    def __init__(self):
        pass

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) 
        h, w, _ = frame.shape
        cv2.imshow('Virtual Mouse', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
                break
    cap.release()
    cv2.destroyAllWindows()