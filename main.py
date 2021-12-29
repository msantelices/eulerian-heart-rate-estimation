import cv2
import os
from model import heartrate, preprocessing, eulerian, pyramids


if not os.path.isdir('temp'):
    os.makedirs('temp')

if not os.path.isdir('result'):
    os.makedirs('result')


# Adjustment
adjust_bpm = 10


# Frequency range for Fast-Fourier Transform
freq_min = 1
freq_max = 1.8

bpm = 0

# Record video
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('temp/capture.avi', fourcc, 20.0, (640, 480))

while(cap.isOpened()):
    ret, frame = cap.read()
    out.write(frame)

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) == 13:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

    
# Preprocessing phase
print("Reading + preprocessing video...")
video_frames, frame_ct, fps = preprocessing.read_video("temp/capture.avi")

# Build Laplacian video pyramid
print("Building Laplacian video pyramid...")
lap_video = pyramids.build_video_pyramid(video_frames)

amplified_video_pyramid = []

for i, video in enumerate(lap_video):
    if i == 0 or i == len(lap_video)-1:
        continue

    # Eulerian magnification with temporal FFT filtering
    print("Running FFT and Eulerian magnification...")
    result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
    lap_video[i] += result

    # Calculate heart rate
    print("Calculating heart rate...")
    bpm = heartrate.find_heart_rate(fft, frequencies, freq_min, freq_max) + adjust_bpm


print("Heart rate: ", bpm, "bpm")



# Output
output = cv2.VideoCapture('temp/capture.avi')
save_output = cv2.VideoWriter('result/result.avi', fourcc, 20.0, (640, 480))
while(output.isOpened()):
    ret, frame = output.read()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        'BPM: ' + str(round(bpm)),
        (50, 50),
        font, 1,
        (0, 255, 255),
        1,
        cv2.LINE_4
    )

    save_output.write(frame)

    cv2.imshow('Output', frame)
    if cv2.waitKey(25) == 13:
        break


output.release()
save_output.release()
cv2.destroyAllWindows()

os.remove('temp/capture.avi')
