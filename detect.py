from ultralytics import YOLO

model = YOLO(r'C:\FootballAnalysis\FrameDetection\Yolo_Training\models\best.pt')

results = model.predict(r'C:\FootballAnalysis\assets\game.mp4', save = True)

print(results[0])
for box in results[0].boxes:
    print(box)