import argparse
import importlib.metadata as metadata
from argparse import Namespace
from datetime import datetime

import cv2

from ocrina.models.log_level import LogLevel
from ocrina.services.recognizer import Recognizer
from ocrina.shared.utils.logger import Logger

__version__ = metadata.version(__package__ or __name__)


class Ocrina:
    def __init__(self):
        self.logger = Logger()
        self.args = self.parse_args()
        self.set_verbosity()
        self.recognizer = Recognizer()

    @staticmethod
    def parse_args() -> Namespace:
        parser = argparse.ArgumentParser(description="This is a template repository to build Python CLI tool.")
        parser.add_argument('--verbose', '-v', action='count', default=1,
                            help='Increase verbosity. Use more than once to increase verbosity level (e.g. -vvv).')
        parser.add_argument('--debug', action='store_true', default=False,
                            help='Enable debug mode.')
        parser.add_argument('--quiet', '-q', action=argparse.BooleanOptionalAction, default=False,
                            required=False, help='Do not print any output/log')
        parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}',
                            help='Show version and exit.')
        parser.add_argument('--on-demand', '-d', action='store_true', default=False,
                            help='Enable on-demand mode.')
        parser.add_argument('--files', '-f', nargs='+', required=False,
                            help='List of files to process.')
        return parser.parse_args()

    def check_args(self) -> None:
        error_message = ""

        if not (bool(self.args.on_demand) ^ bool(self.args.files)):
            error_message = "You must provide either --on-demand or --files argument."

        if error_message != "":
            self.logger.error(error_message)
            exit(1)

    def set_verbosity(self) -> None:
        if self.args.quiet:
            verbosity_level = LogLevel.DISABLED
        else:
            if self.args.debug or self.args.verbose > LogLevel.DEBUG.value:
                verbosity_level = LogLevel.DEBUG
            else:
                verbosity_level = self.args.verbose
        self.logger.set_log_level(verbosity_level)

    def run(self):
        self.check_args()
        self.logger.info(f"Running...")
        self.logger.debug(self.args)
        if self.args.files:
            self.recognize_files()
        elif self.args.on_demand:
            self.on_demand()
        else:
            self.logger.error("Invalid arguments.")
            exit(1)

    def recognize_files(self):
        for file in self.args.files:
            self.logger.info(f"Processing file: {file}")
            text = self.recognizer.recognize_file(file)
            self.logger.info(f"Text: {text}")

    def on_demand(self):
        self.logger.info("Running in on-demand mode.")
        width = 2560
        height = 1440
        fps = 30
        capture_format = 'MJPG'
        esc_key = 27
        exit_key = esc_key
        pooling_seconds = 1
        last_pooling_time = datetime.now()

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left_corner_of_text = (40, 40)
        font_scale = 2
        font_color = (255, 255, 255)
        thickness = 1
        line_type = 2

        cv2.namedWindow("preview")
        vc = cv2.VideoCapture(0)

        # vc.set(cv2.CAP_PROP_FPS, fps)
        # vc.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # vc.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        vc.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*capture_format))

        frame = None
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False

        label = "Detecting..."
        while rval:
            cv2.putText(frame, label, bottom_left_corner_of_text, font, font_scale, font_color, thickness, line_type)
            cv2.imshow("preview", frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)
            if key == exit_key:
                break
            if (datetime.now() - last_pooling_time).seconds > pooling_seconds:
                label = "Recognizing..."
                text = self.recognizer.recognize_raw_image(frame)
                print("Text: ", text)
                if text is not None and len(text) > 0:
                    self.logger.info(f"Text: {text}")
                last_pooling_time = datetime.now()

        vc.release()
        cv2.destroyWindow("preview")
