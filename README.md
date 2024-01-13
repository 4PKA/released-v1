# Прогнозирование времени прибытия с использованием разпознавания лиц

## Введение
Проект "Прогнозирование времени прибытия" представляет собой систему на языке программирования Python, которая использует технологию разпознавания лиц для прогнозирования времени прихода студентов в учебную аудиторию.

## Компоненты проекта

1. **Библиотеки и инструменты**
   - **OpenCV:** Используется для обработки изображений и видеопотока.
   - **Dlib:** Включает в себя инструменты для детекции лиц, извлечения признаков и разпознавания лиц.
   - **NumPy:** Используется для работы с массивами и вычислений.
   - **Threading:** Многопоточность для одновременной обработки видеокадров и управления интерфейсом.

2. **База данных PostgreSQL**
   - Содержит таблицу `faces` с информацией о студентах, включая их имена, фамилии, группы и пути к фотографиям.

3. **CSV-файл "опоздавшие_студенты.csv"**
   - Файл, в который записываются данные о студентах, показавших свои лица и прогнозе времени прибытия.

4. **Алгоритм работы программы**
   - Детекция лиц с использованием библиотеки Dlib.
   - Извлечение признаков лица для каждого обнаруженного лица.
   - Прогнозирование времени прибытия студента на основе данных о распознанном лице.
   - Визуализация результата на видеопотоке с отображением имени студента и прогнозируемого времени прибытия.

## Запуск проекта

1. **Настройка окружения**
   - Установите необходимые библиотеки, используя `pip install opencv-python dlib numpy psycopg2`.

2. **Подготовка базы данных**
   - Создайте базу данных PostgreSQL с таблицей `faces`, содержащей информацию о студентах.
   - Заполните таблицу данными о студентах, включая пути к их фотографиям.

3. **Запуск программы**
   - Запустите скрипт с помощью Python: `python your_script_name.py`.
   - В результате запуска программа начнет обработку видеопотока, разпознавая лица и прогнозируя время прибытия студентов.

## Важно
- Убедитесь, что все необходимые библиотеки установлены перед запуском.
- При необходимости измените параметры базы данных (хост, имя, пользователь, пароль) в соответствии с вашим окружением в коде.

**Эта документация предоставляет общий обзор проекта. Перед использованием удостоверьтесь, что ваше окружение настроено правильно, и выполните необходимые шаги для подготовки базы данных и CSV-файла.**
