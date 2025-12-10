import cv2


def test_http(http_url: str):
    cap = cv2.VideoCapture(http_url)

    if not cap.isOpened():
        print("Error: Could not open stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
