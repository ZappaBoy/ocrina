import cv2
import easyocr
from cv2 import Mat, UMat
from numpy import ndarray

from ocrina.shared.utils.logger import Logger

type RawImage = Mat | ndarray | UMat


class Recognizer:
    def __init__(self, languages: list[str] = None, use_gpu: bool = False):
        self.logger = Logger()
        if languages is None:
            languages = ['en']
        self.reader = easyocr.Reader(languages, gpu=use_gpu)

    def recognize_image(self, image: RawImage) -> str:
        result = self.reader.readtext(image)
        print(result)
        text = ""
        for (bbox, recognized_text, prob) in result:
            text += f"{recognized_text} "
        return text

    def recognize_file(self, path: str) -> str:
        raw_image = cv2.imread(path)
        text = self.recognize_image(raw_image)
        return text

