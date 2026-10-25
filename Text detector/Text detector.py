import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def split_box_by_projection(letter_mask, x, y, w, h, min_gap_len=4, col_thr_ratio=0.08):
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

def find_and_draw_letters(letter_mask, final_img, bx, by, bw, bh, offset_x, offset_y):
    """
    Ищет контуры отдельных букв внутри найденной области слова и рисует их красным.
    """
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

            # Рисуем рамку буквы (красным цветом, толщина 1)
            cv2.rectangle(final_img, (abs_x, abs_y), (abs_x + lw, abs_y + lh), (0, 0, 255), 1)

def detect_text(image_path):

    img = cv2.imread(image_path)
    if img is None:
        print("Не прошла детекцию, ошибка с изображение")
        return

    roi = cv2.selectROI("Selection", img, False)
    cv2.destroyWindow("Selection")

    if roi[2] == 0 or roi[3] == 0:
        print("Детекция отменена.")
        return 4

    x_r, y_r, w_r, h_r = roi
    if w_r == 0 or h_r == 0: return

    crop = img[y_r:y_r+h_r, x_r:x_r+w_r]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    p = 25
    gray = cv2.copyMakeBorder(gray, p, p, p, p, cv2.BORDER_CONSTANT, value=0)

    # Инверсия (если фон светлее текста) — критично для "Лета"
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
                        find_and_draw_letters(letter_mask, final_img, sx, sy, sw, sh, offset_x, offset_y)
            else:
                # 1. Рисуем зеленую рамку для слова/фразы (толщина 2)
                cv2.rectangle(final_img, (offset_x + x, offset_y + y),
                              (offset_x + x + w, offset_y + y + h), (0, 255, 0), 2)
                # 2. Ищем и рисуем красные рамки для БУКВ внутри
                find_and_draw_letters(letter_mask, final_img, x, y, w, h, offset_x, offset_y)

    cv2.imshow(" ", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Скрываем главное окно tkinter
root = tk.Tk()
root.withdraw()

# Открываем диалог выбора файла
file_path = filedialog.askopenfilename(
    title="Выберите файл CSV",
    filetypes=[("Image files", "*.png *.jpeg *.jpg")]
)

print("Выбран файл:", file_path)
detect_text(file_path)