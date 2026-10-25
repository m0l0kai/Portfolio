# .\venv311\Scripts\pyside6-designer.exe
from PySide6.QtGui import QPainter, QPen, QColor, QGuiApplication, QImage
from PySide6.QtWidgets import QApplication, QPushButton, QLabel
from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QWidget
import Search_text as sr
import translators as ts
import numpy as np
import easyocr
import cv2
import warnings

warnings.filterwarnings("ignore",  message=".*CUDA.*")
from spellchecker import SpellChecker

# Инициализируем один раз при запуске скрипта
spell = SpellChecker(language='en')

# добавляем в словарь сокращения, которые spellchecker может не знать,
# чтобы он случайно не "исправил" их на обычные слова
tech_exceptions = ['rpm', 'sa', 'abs', 'spd', 'gear', 'eng']
spell.word_frequency.load_words(tech_exceptions)

class OCR:
    _reader = None

    @staticmethod
    def get_reader():
        if OCR._reader is None:
            OCR._reader = easyocr.Reader(['en'], gpu=False)
        return OCR._reader

class Overlay(QWidget):
    # Создаем кастомный сигнал, который будет передавать саму рамку (object)
    closed_signal = Signal(object)

    def __init__(self, parent_window=None):
        super().__init__()
        self.main_win = parent_window
        self.setMouseTracking(True)
        self._resize_dir = None

        # Окно без рамок, поверх всех, прозрачный фон
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(400, 250)

        self.locked = False
        self.drag_pos = QPoint()
        self.resizing = False
        self.resize_margin = 10

        # Флаг для управления рамкой в скриншоте
        self.show_border = True

        # Кнопки
        self.btn_pin = QPushButton("📌", self)
        self.btn_pin.setGeometry(5, 5, 25, 25)
        self.btn_pin.clicked.connect(self.toggle_lock)
        # Принудительно ставим стрелку, чтобы Move-курсор не перекрывал
        self.btn_pin.setCursor(Qt.ArrowCursor)

        self.btn_close = QPushButton("✕", self)
        self.btn_close.clicked.connect(self.destroy_overlay)
        self.btn_close.setGeometry(35, 5, 25, 25)
        self.btn_close.clicked.connect(self.close)
        # Принудительно ставим стрелку
        self.btn_close.setCursor(Qt.ArrowCursor)

        self.setStyleSheet("""
            QPushButton { background: rgba(40,40,40,140); color: white; border-radius: 6px; }
            QPushButton:hover { background: rgba(80,80,80,200); }
        """)

        self.label_rez = QLabel(self)
        self.label_rez.setAlignment(Qt.AlignCenter)
        self.label_rez.setWordWrap(True)  # Чтобы текст переносился
        self.label_rez.setVisible(False)  # Изначально скрыта

        # Стили метки: полупрозрачный фон и яркий текст
        self.label_rez.setStyleSheet("""
                    QLabel {
                        background-color: rgba(0, 0, 0, 180); 
                        color: #00FF00; 
                        font-size: 16px; 
                        font-weight: bold;
                        border-radius: 4px;
                    }
                """)

        self.i_dop = 0 # для проверок/оценки
        self.reader = OCR.get_reader()

    def destroy_overlay(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()  # Останавливаем таймер этой конкретной рамки

        self.close()  # Закрываем виджет
        self.deleteLater()  # Гарантированно уничтожаем объект в Qt

    def toggle_lock(self):
        self.locked = not self.locked
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)

        # Заливка для кликабельности (невидимая)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 1))

        # Рисуем рамку только если флаг включен
        if self.show_border:
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

    def get_resize_direction(self, pos):
        if self.locked: return None
        rect = self.rect()
        l = pos.x() < self.resize_margin
        r = pos.x() > rect.width() - self.resize_margin
        t = pos.y() < self.resize_margin
        b = pos.y() > rect.height() - self.resize_margin

        if t and l: return "tl"
        if t and r: return "tr"
        if b and l: return "bl"
        if b and r: return "br"
        if l: return "l"
        if r: return "r"
        if t: return "t"
        if b: return "b"
        return None

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()

        # Установка курсора (только если не ресайзим)
        if not self.resizing and not self.locked:
            # Проверяем, находится ли курсор НАД КНОПКОЙ
            # childAt() возвращает виджет по этим координатам
            child_widget = self.childAt(pos)

            # Если под мышкой нет кнопки (или это сам Overlay)
            if not child_widget or child_widget == self:
                direction = self.get_resize_direction(pos)
                if direction in ("l", "r"):
                    self.setCursor(Qt.SizeHorCursor)
                elif direction in ("t", "b"):
                    self.setCursor(Qt.SizeVerCursor)
                elif direction in ("tl", "br"):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif direction in ("tr", "bl"):
                    self.setCursor(Qt.SizeBDiagCursor)
                else:
                    # Центральная часть для перемещения
                    self.setCursor(Qt.SizeAllCursor)
            else:
                # Если курсор над кнопкой, возвращаем стрелку
                self.setCursor(Qt.ArrowCursor)

        if event.buttons() & Qt.LeftButton and not self.locked:
            new_global_pos = event.globalPosition().toPoint()
            delta = new_global_pos - self.drag_pos
            geom = self.geometry()
            x, y, w, h = geom.x(), geom.y(), geom.width(), geom.height()

            if self.resizing:
                if "r" in self._resize_dir: w += delta.x()
                if "l" in self._resize_dir:
                    diff = delta.x()
                    if w - diff > 100: x += diff; w -= diff
                if "b" in self._resize_dir: h += delta.y()
                if "t" in self._resize_dir:
                    diff = delta.y()
                    if h - diff > 80: y += diff; h -= diff
                self.setGeometry(x, y, max(100, w), max(80, h))
            else:
                self.move(self.pos() + delta)

            self.drag_pos = new_global_pos

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Запоминаем локальные координаты клика
            self.drag_pos = event.globalPosition().toPoint()

            # Логика сжатого кулака при перемещении
            pos = event.position().toPoint()
            child_widget = self.childAt(pos)

            self._resize_dir = self.get_resize_direction(pos)
            self.resizing = self._resize_dir is not None

            # Если зажали по центру (не ресайз) и не на кнопке
            if not self.resizing and (not child_widget or child_widget == self):
                self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        self.resizing = False
        self._resize_dir = None

        # Обновляем курсор сразу после отпускания (чтобы вернулась ладонь/ресайз)
        self.mouseMoveEvent(event)

    def translation(self, letters):

        word = ''.join(letters)

        if not word.strip():
            return "Текст не найден"

        print('DO translation', word)

        mode = self.main_win.comboBox.currentText()

        if mode == "Яндекс":
            rez = ts.translate_text(word, translator='yandex', to_language='ru')
            print("Яндекс")
        elif mode == "Google":
            rez = ts.translate_text(word, translator='google', to_language='ru')
            print("Google")
        else:
            rez = ts.translate_text(word, translator='bing', to_language='ru')
            print("Bing")

        print('POSLE translation', rez)
        print('-'*20)
        return rez

    def capture_area_clean(self):
        """Делает чистый скриншот без UI"""

        # Скрываем кнопки
        self.btn_pin.hide()
        self.btn_close.hide()

        # Выключаем рамку в paintEvent и обновляем окно
        self.show_border = False
        self.update()

        # Даем Qt время полностью перерисовать окно перед захватом
        QApplication.processEvents()

        # Делаем захват области
        geo = self.geometry()
        screen = QGuiApplication.primaryScreen()
        if not screen:
            self._restore_ui()  # Если экрана нет, аварийное восстановление
            return None

        pixmap = screen.grabWindow(0, geo.x(), geo.y(), geo.width(), geo.height())

        # МОСТ В OPENCV
        image = pixmap.toImage()
        image = image.convertToFormat(QImage.Format_RGB888)

        width = image.width()
        height = image.height()
        bytes_per_line = image.bytesPerLine()  # Получаем реальный шаг строки

        ptr = image.bits()

        # Создаем массив, учитывая реальный размер байт в строке
        # Мы берем данные как одномерный массив и превращаем в матрицу
        arr = np.frombuffer(ptr, np.uint8).reshape((height, bytes_per_line))

        # Отрезаем лишние байты (отступы) в конце каждой строки,
        # оставляя только данные пикселей (width * 3 канала)
        arr = arr[:, :width * 3].reshape((height, width, 3))

        # Конвертируем из RGB (Qt) в BGR (OpenCV)
        opencv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # Восстанавливаем UI
        self._restore_ui()

        # Отправляем наш вырезанный кусок в детектор
        img, letters_mass = sr.detect_text(opencv_img, self.i_dop)
        self.i_dop += len(img)

        final_sentences = []

        for word_img in img:

            h, w = word_img.shape[:2]

            # Говорим EasyOCR: "Весь этот кусок — это один текстовый бокс, читай от края до края!"
            # Формат EasyOCR: [x_min, x_max, y_min, y_max]
            box_full_image = [0, w, 0, h]

            # Используем recognize, чтобы пропустить этап внутренней детекции
            res = self.reader.recognize(
                word_img,
                horizontal_list=[box_full_image],
                free_list=[],
                detail=1,
                decoder='greedy'
            )
            if res:
                if res[0][2] > 0.20:
                    print(res[0][1])
                    final_sentences.append(res[0][1])

        # Объединяем всё в финальный текст интерфейса
        print("Распознанный текст:", " ".join(final_sentences))

        # Переводим готовую "ленту" символов
        rez = self.translation(" ".join(final_sentences))

        # Записываем текст в метку
        self.label_rez.setText(rez)

        # Растягиваем метку на всё окно
        self.label_rez.setGeometry(0, 0, self.width(), self.height())

        # Показываем результат
        self.label_rez.setVisible(True)

        # Скрываем кнопки управления, чтобы они не мешали читать
        self.btn_pin.hide()
        self.btn_close.hide()
        self.show_border = False  # Прячем красную рамку
        self.update()


    def _restore_ui(self):
        """Возвращает кнопки и красную рамку"""
        self.btn_pin.show()
        self.btn_close.show()
        self.show_border = True
        self.update()

    def resizeEvent(self, event):
        self.label_rez.setGeometry(0, 0, self.width(), self.height())
        super().resizeEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Возвращает рамку в обычный режим при двойном клике"""
        if self.label_rez.isVisible():
            self.label_rez.setVisible(False)
            self._restore_ui()  # Показываем кнопки и красную рамку обратно

    def closeEvent(self, event):
        # Этот код сработает автоматически при ЛЮБОМ закрытии окна
        print("Оверлей перехватил событие закрытия окна")

        # Отправляем сигнал главному окну, чтобы оно вычеркнуло нас из overlays
        self.closed_signal.emit(self)

        # Останавливаем таймеры, если они есть
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()

        # Разрешаем окну закрыться
        event.accept()

if __name__ == "__main__":

    app = QApplication([])
    loader = QUiLoader()
    window = loader.load("main.ui")

    overlays = []

    # Функция-помощник для удаления мертвой рамки из списка
    def remove_dead_overlay(overlay_obj):
        if overlay_obj in overlays:
            overlays.remove(overlay_obj)
            print(f"Рамка успешно удалена из памяти. Осталось активных зон: {len(overlays)}")

    def Create():
        # Создаем новый экземпляр рамки
        new_rect = Overlay(parent_window=window)

        # СВЯЗЫВАЕМ СИГНАЛ: когда рамка закроется, вызовется функция удаления из списка
        new_rect.closed_signal.connect(remove_dead_overlay)

        # Смещаем новое окно, чтобы они не накладывались ровно друг на друга
        offset = len(overlays) * 30
        new_rect.move(100 + offset, 100 + offset)

        # Показываем окно
        new_rect.show()

        # Сохраняем ссылку в список
        overlays.append(new_rect)

        print(f"Создана рамка №{len(overlays)}")


    def on_start():
        if not overlays:
            print("Нет ни одной активной рамки для сканирования!")
            return

        print(f"Запуск сканирования для {len(overlays)} зон...")

        # Итерируемся по КОПИИ списка, чтобы удаление элементов не сбивало цикл
        for zone in list(overlays):
            try:
                zone.capture_area_clean()
            except RuntimeError:
                print("Предупреждение: Обнаружена устаревшая зона, подчищаем память.")
                # Безопасно удаляем из оригинального списка
                if zone in overlays:
                    overlays.remove(zone)

        print(f"Сканирование завершено. Активных зон осталось: {len(overlays)}")

    # Привязываем к кнопкам из вашего UI
    window.btnStart.clicked.connect(on_start)
    window.btnCreate.clicked.connect(Create)

    window.show()
    app.exec()