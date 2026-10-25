import numpy as np
import cv2


import shutil
import os
def save(img, letters_mass, name):
    folder_path = name
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)
    target_h = 64
    j = 1
    for i in letters_mass:
        x1, y1, x2, y2 = i[0], i[1], i[2], i[3]

        # Вырезаем строго по контуру буквы из оригинального цветного img
        cropped = img[y1:y2, x1:x2]

        h, w = cropped.shape[:2]
        if h < target_h:
            new_h = target_h
            # Формула пропорционального расчета ширины
            new_w = int(round(w * (new_h / h)))

            # Изменяем размер с использованием кубической интерполяции для сохранения четкости
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        prepared_crop = preprocess_for_ocr(cropped)

        # Добавляем идеально подготовленную цветную букву в массив для EasyOCR
        cv2.imwrite(f'{name}/{j}.png', prepared_crop)
        j+=1


def preprocess_for_ocr(cropped_image):
    """
    Превращает крошечный вырезанный силуэт буквы в четкий крупный черно-белый трафарет.
    """
    if cropped_image is None or cropped_image.size == 0:
        return cropped_image

    # Переводим в ч/б (если картинка вдруг цветная)
    if len(cropped_image.shape) == 3:
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped_image

    # EasyOCR создавался для чтения вывесок и номеров, мелкие пиксели он не видит.
    resized = cv2.resize(gray, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)

    # Сглаживаем пиксели после увеличения, чтобы края не были "лесенкой"
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    # Бинаризация Оцу: автоматически делает фон идеально белым, а букву - черной
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # EasyOCR лучше всего читает ЧЕРНЫЙ текст на БЕЛОМ фоне.
    # Если у тебя в игре буквы светлые, а фон темный — инвертируем картинку.
    white_pixels = cv2.countNonZero(binarized)
    total_pixels = binarized.shape[0] * binarized.shape[1]
    if white_pixels < (total_pixels / 2):
        binarized = cv2.bitwise_not(binarized)

    return binarized


def sort_blocks(blocks):
    if not blocks:
        return []

    # ычисляем центр Y и высоту для каждого бокса
    extended_blocks = []
    for b in blocks:
        cy = (b[1] + b[3]) / 2.0  # Центр буквы по вертикали
        h = b[3] - b[1]  # Высота буквы
        extended_blocks.append((b, cy, h))

    # Предварительно сортируем элементы сверху вниз по их центру Y
    extended_blocks.sort(key=lambda x: x[1])

    lines = []  # Здесь будем хранить сгруппированные строки

    for b, cy, h in extended_blocks:
        placed = False

        # Пытаемся найти уже существующую строку, в которую впишется этот блок
        for line in lines:
            # Считаем средний центр Y и среднюю высоту для текущей строки
            line_cy = sum(item[1] for item in line) / len(line)
            line_h = sum(item[2] for item in line) / len(line)

            # Допуск (0.6 * среднюю высоту букв в строке).
            # Это позволяет легко ловить "улетающие" вверх тире и падающие вниз хвосты 'y', 'g', 'p'
            if abs(cy - line_cy) < (line_h * 0.6):
                line.append((b, cy, h))
                placed = True
                break

        # Если ни одна строка не подошла — создаем новую строку
        if not placed:
            lines.append([(b, cy, h)])

    # Сортируем сами строки сверху вниз по их реальному расположению
    lines.sort(key=lambda line: sum(item[1] for item in line) / len(line))

    # Внутри каждой строки сортируем символы слева направо (по координате X)
    sorted_blocks = []
    for line in lines:
        line.sort(key=lambda item: item[0][0])  # Сортировка по x1
        for item in line:
            sorted_blocks.append(item[0])  # Возвращаем чистый исходный бокс

    return sorted_blocks

def split_box_by_projection(letter_mask, x, y, w, h, min_gap_len=10, col_thr_ratio=0.08):
    """
    Делит широкий бокс по вертикальной проекции.
    Возвращает список боксов [(x, y, w, h), ...] относительно letter_mask.
    """
    roi = letter_mask[y:y + h, x:x + w]
    if roi.size == 0:
        return [(x, y, w, h)]

    # Сколько белых пикселей в каждом столбце
    proj = np.sum(roi > 0, axis=0)
    col_thr = max(1, int(h * col_thr_ratio))

    cuts = []
    in_gap = False
    gap_start = 0

    for i, v in enumerate(proj):
        if v <= col_thr:
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                gap_len = i - gap_start
                if gap_len >= min_gap_len:
                    cuts.append((gap_start + i) // 2)
                in_gap = False

    if not cuts:
        return [(x, y, w, h)]

    parts = []
    start = 0
    for c in cuts:
        if c - start > 5:
            parts.append((x + start, y, c - start, h))
        start = c

    if w - start > 5:
        parts.append((x + start, y, w - start, h))

    return parts

def analysis(img,i_dop):
    for i in range(len(img)):
        cv2.imwrite(f'search/{i + i_dop}.png', img[i])

def assemble_boxes_into_words(blocks, x_gap_coef=0.8, y_overlap_coef=0.4):
    """
    Собирает отдельные боксы букв в целые слова на основе геометрического расположения.

    :param blocks: Список исходных боксов букв [[x1, y1, x2, y2], ...]
    :param x_gap_coef: Коэффициент чувствительности к пробелам (чем больше, тем охотнее сшивает буквы)
    :param y_overlap_coef: Какой процент высоты буквы должны пересекаться по вертикали, чтобы быть в одной строке
    :return: Список боксов готовых слов [[x1, y1, x2, y2], ...]
    """
    if not blocks:
        return []

    # Принудительно переводим во внутренний формат списков для изменяемости
    letter_boxes = [list(b) for b in blocks]

    # ГРУППИРОВКА БУКВ ПО СТРОКАМ
    # Сортируем все буквы сверху вниз по координате Y1
    letter_boxes.sort(key=lambda b: b[1])

    lines = []
    current_line = [letter_boxes[0]]

    for i in range(1, len(letter_boxes)):
        prev_b = current_line[-1]
        curr_b = letter_boxes[i]

        prev_h = prev_b[3] - prev_b[1]
        curr_h = curr_b[3] - curr_b[1]

        # Вычисляем, насколько буквы пересекаются друг под другом по вертикали
        overlap_y = min(prev_b[3], curr_b[3]) - max(prev_b[1], curr_b[1])
        min_h = min(prev_h, curr_h)

        # Если есть хорошее пересечение по Y или их верхние границы почти на одной линии
        if overlap_y > (min_h * y_overlap_coef) or abs(curr_b[1] - prev_b[1]) < (min_h * 0.3):
            current_line.append(curr_b)
        else:
            # Обнаружена новая строка
            lines.append(current_line)
            current_line = [curr_b]

    # Не забываем добавить последнюю строку
    lines.append(current_line)

    # СБОРКА БУКВ В СЛОВА ВНУТРИ КАЖДОЙ СТРОКИ
    words_mass = []

    for line in lines:
        # Сортируем буквы внутри строки строго СЛЕВА НАПРАВО по координате X1
        line.sort(key=lambda b: b[0])

        # Берем первую букву строки за основу текущего слова
        current_word = list(line[0])

        for i in range(1, len(line)):
            curr_b = line[i]

            # Считаем реальный зазор по горизонтали (X) между словом и следующей буквой
            gap_x = curr_b[0] - current_word[2]

            # Метрики текущей буквы для адаптивного порога
            char_h = curr_b[3] - curr_b[1]
            char_w = curr_b[2] - curr_b[0]

            # АДАПТИВНЫЙ ПОРОГ ПРОБЕЛА:
            # Зависит от высоты и ширины буквы. Для крупного шрифта порог растет, для мелкого — уменьшается.
            max_gap_x = max(char_h * x_gap_coef, char_w * 0.9)

            # Если буква стоит близко к слову — поглощаем её (склеиваем)
            if gap_x < max_gap_x:
                current_word[0] = min(current_word[0], curr_b[0])  # Новый левый край
                current_word[1] = min(current_word[1], curr_b[1])  # Новый верхний край
                current_word[2] = max(current_word[2], curr_b[2])  # Новый правый край
                current_word[3] = max(current_word[3], curr_b[3])  # Новый нижний край
            else:
                # Порог превышен -> это пробел между словами!
                # Сохраняем собранное слово и начинаем новое с текущей буквы
                words_mass.append(current_word)
                current_word = list(curr_b)

        # Сохраняем последнее слово строки
        words_mass.append(current_word)

    return words_mass


def merge_i_dots(blocks):
    """
    Склеивает точки над 'i', 'j' и под '!', '?' с их базами.
    Включает защиту от межстрочного склеивания точек препинания.
    """
    if not blocks:
        return []

    # Вычисляем медианную высоту букв, чтобы понимать общий масштаб текста
    heights = [b[3] - b[1] for b in blocks]
    median_h = np.median(heights) if heights else 20

    merged = []
    used = set()

    # Сортируем боксы сверху вниз по оси Y
    blocks = sorted(blocks, key=lambda x: x[1])

    for i in range(len(blocks)):
        if i in used:
            continue

        b1 = blocks[i]
        x1_1, y1_1, x2_1, y2_1 = b1
        h1 = y2_1 - y1_1
        w1 = x2_1 - x1_1
        cx1 = (x1_1 + x2_1) / 2.0

        saved_box = list(b1)

        for j in range(i + 1, len(blocks)):
            if j in used:
                continue

            b2 = blocks[j]
            x1_2, y1_2, x2_2, y2_2 = b2
            h2 = y2_2 - y1_2
            w2 = x2_2 - x1_2
            cx2 = (x1_2 + x2_2) / 2.0

            # Центры объектов по оси X должны быть очень близки
            max_w = max(w1, w2)
            if abs(cx1 - cx2) > (max_w * 0.6):
                continue

            # Определяем, кто из них точка, а кто база (палочка)
            if h1 < h2:
                dot_h, dot_w = h1, w1
                base_h, base_w = h2, w2
            else:
                dot_h, dot_w = h2, w2
                base_h, base_w = h1, w1

            # Фильтр формы: Точка должна быть реально маленькой!
            if dot_h > (median_h * 0.4) or dot_w > (median_h * 0.4):
                continue  # Оба элемента большие, склейка отменяется

            # Вычисляем физический зазор между ними по оси Y
            # Так как мы отсортировали массив, b1 всегда выше (или на уровне) b2
            gap_y = y1_2 - y2_1

            # ГЛАВНЫЙ ЩИТ ОТ СКЛЕЙКИ СТРОК:
            # Зазор должен быть пропорционален самой точке!
            # Допускаем зазор не больше, чем 2.5 высоты точки.
            if gap_y > (dot_h * 2.5) or gap_y < -5:
                continue

            # Если все проверки пройдены - это 100% буква i, j, ! или ?
            saved_box[0] = min(saved_box[0], x1_2)
            saved_box[1] = min(saved_box[1], y1_2)
            saved_box[2] = max(saved_box[2], x2_2)
            saved_box[3] = max(saved_box[3], y2_2)
            used.add(j)
            break  # Склеили - переходим к следующему объекту

        merged.append(saved_box)

    return merged

def find_and_draw_letters(letter_mask, final_img, bx, by, bw, bh, offset_x, offset_y):
    """
    Ищет контуры отдельных букв внутри найденной области слова и рисует их красным.
    """
    letters_mass = []
    # Вырезаем область из маски, где буквы еще не склеены
    roi_mask = letter_mask[by:by+bh, bx:bx+bw]

    # Ищем контуры (отдельные буквы)
    l_cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in l_cnts:
        lx, ly, lw, lh = cv2.boundingRect(c)
        # Фильтруем совсем мелкий пиксельный мусор (точки)
        if lw > 2 and lh > 4:
            # Вычисляем глобальные координаты на оригинальном изображении
            abs_x = offset_x + bx + lx
            abs_y = offset_y + by + ly

            letters_mass.append([abs_x, abs_y, abs_x + lw, abs_y + lh])

            # Рисуем рамку буквы (красным цветом, толщина 1)
            cv2.rectangle(final_img, (abs_x, abs_y), (abs_x + lw, abs_y + lh), (0, 0, 255), 1)

    return letters_mass

def detect_text(img, i_dop):

    img_mas = []

    if img is None:
        print("Не прошла детекцию, ошибка с изображение")
        return

    h_r, w_r = img.shape[:2]
    x_r = 0
    y_r = 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p = 25
    gray = cv2.copyMakeBorder(gray, p, p, p, p, cv2.BORDER_CONSTANT, value=0)

    # Инверсия (если фон светлее текста)
    if np.mean(gray) > 120:
        gray = cv2.bitwise_not(gray)

    letter_mask = np.zeros_like(gray)
    final_img = img.copy()

    # Выделение символьных областей
    # Здесь используем MSER, так как текст контрастный, но фон сложный
    # Убираем мелкий шум фона, сохраняя буквы
    blur = cv2.bilateralFilter(gray, 2, 75, 75)

    mser = cv2.MSER_create(min_area=15, max_area=int((w_r * h_r) * 0.05), delta=8)
    regions, _ = mser.detectRegions(blur)

    for r in regions:
        # Геометрическая фильтрация прямо на этапе маски
        bx, by, bw, bh = cv2.boundingRect(r)
        if 0.2 < (bw/bh) < 5.0 and bh > 5: # Буквы обычно имеют вменяемые пропорции
            cv2.drawContours(letter_mask, [r], -1, 255, -1)

    k_w_glue = 25

    # --- УНИВЕРСАЛЬНАЯ СКЛЕЙКА И ОТРИСОВКА ---
    line_glue = cv2.dilate(letter_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k_w_glue, 1)))
    f_cnts, _ = cv2.findContours(line_glue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letters_mass = []

    # Заранее считаем смещение, чтобы вернуть координаты на оригинальное фото
    offset_x = x_r - p
    offset_y = y_r - p

    for c in f_cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 8 and h > 6 and w < gray.shape[1] * 0.96:
            roi_bits = letter_mask[y:y+h, x:x+w]
            white_ratio = np.count_nonzero(roi_bits) / float(w * h + 1e-9)

            # Если в боксе слишком мало "буквенных" пикселей - это шум
            if white_ratio < 0.015: continue

            if w > 160 or (w/h) > 6.0:
                sub_boxes = split_box_by_projection(letter_mask, x, y, w, h)
                for sx, sy, sw, sh in sub_boxes:
                    if sw > 6:
                        # 1. Рисуем зеленую рамку для слова/фразы (толщина 2)
                        cv2.rectangle(final_img, (offset_x + sx, offset_y + sy),
                                      (offset_x + sx + sw, offset_y + sy + sh), (0, 255, 0), 2)
                        # 2. Ищем и рисуем красные рамки для БУКВ внутри
                        xy_xy = find_and_draw_letters(letter_mask, final_img, sx, sy, sw, sh, offset_x, offset_y)
                        letters_mass.extend(xy_xy)

            else:
                # Рисуем зеленую рамку для слова/фразы (толщина 2)
                cv2.rectangle(final_img, (offset_x + x, offset_y + y),
                              (offset_x + x + w, offset_y + y + h), (0, 255, 0), 2)
                # Ищем и рисуем красные рамки для БУКВ внутри
                xy_xy = find_and_draw_letters(letter_mask, final_img, x, y, w, h, offset_x, offset_y)
                letters_mass.extend(xy_xy)

    cv2.imwrite(f'search/orig_{i_dop}.png', final_img)

    print("Нашел: ", len(letters_mass))
    letters_mass = sort_blocks(letters_mass)
    save(img, letters_mass, "sort")
    print("Сортировка: ", len(letters_mass))
    letters_mass = merge_i_dots(letters_mass)
    print("Исправление: ",len(letters_mass))
    letters_mass = sort_blocks(letters_mass)
    save(img, letters_mass, "merge")
    print("Сортировка: ", len(letters_mass))
    letters_mass = assemble_boxes_into_words(letters_mass)
    save(img, letters_mass, "assemble")
    print("Соединено: ", len(letters_mass))

    target_h = 64
    for i in letters_mass:
        x1, y1, x2, y2 = i[0], i[1], i[2], i[3]

        # Вырезаем строго по контуру буквы из оригинального цветного img
        cropped = img[y1:y2, x1:x2]

        h, w = cropped.shape[:2]
        if h < target_h:
            new_h = target_h
            # Формула пропорционального расчета ширины
            new_w = int(round(w * (new_h / h)))

            # Изменяем размер с использованием кубической интерполяции для сохранения четкости
            cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        prepared_crop = preprocess_for_ocr(cropped)

        # Добавляем идеально подготовленную цветную букву в массив для EasyOCR
        img_mas.append(prepared_crop)

    analysis(img_mas, i_dop)

    return img_mas, letters_mass
#     print("==============================================")
#     summ += len(img)
#     # print(len(img))

