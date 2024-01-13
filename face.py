import cv2
import dlib
import numpy as np
import threading
import psycopg2
import csv
from datetime import datetime
import os

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') #Скачайте этот файл вручную, я бы вложил этот файл в свой проект но он слишком много весит :)
face_rec = dlib.face_recognition_model_v1('путь\\к\\файлу\\dlib_face_recognition_resnet_model_v1.dat')

conn = psycopg2.connect(
    host="your_database_host",
    database="your_database_name",
    user="your_database_user",
    password="your_database_password",
    port=your_database_port
)

cursor = conn.cursor()

cursor.execute("SELECT id, name, surname, n_group, photo_path FROM faces;")
records = cursor.fetchall()

your_faces = {}
for record in records:
    id, name, surname, n_group, photo_path = record
    img = cv2.imread(photo_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    shape = predictor(gray, faces[0])
    descriptor = np.array(face_rec.compute_face_descriptor(img, shape))
    your_faces[name] = {
        'surname': surname,
        'n_group': n_group,
        'descriptor': descriptor,
    }

cursor.close()
conn.close()

# Открытие видеопотока
cap = cv2.VideoCapture(0)

# Имя CSV-файла
csv_file_path = 'опоздавшие_студенты.csv'

fieldnames = ['Имя', 'Фамилия', 'Группа', 'Время опоздания']

# Проверка наличия файла и создание, если он не существует
if not os.path.isfile(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

with open(csv_file_path, mode='r') as csv_file:
    reader = csv.DictReader(csv_file)
    existing_names = set(row['Имя'] for row in reader)

exit_flag = False

def recognize_face(frame, shape, current_face_descriptor):
    for your_face in your_faces:
        descriptor = your_faces[your_face]['descriptor']
        if np.linalg.norm(current_face_descriptor - descriptor) < 0.5:
            return your_face
    return "unknown"

def process_frames():
    global exit_flag
    while not exit_flag:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            current_face_descriptor = face_rec.compute_face_descriptor(frame, shape)

            name = recognize_face(frame, shape, current_face_descriptor)

            if name != "unknown" and name not in existing_names:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(csv_file_path, mode='a', newline='') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writerow({
                        'Имя': name + ' ' + your_faces[name]['surname'],
                        'Группа': your_faces[name]['n_group'],
                        'Время опоздания': current_time
                    })
                existing_names.add(name)

            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True
            break
processing_thread = threading.Thread(target=process_frames)
processing_thread.start()

processing_thread.join()

cap.release()
cv2.destroyAllWindows()