import os
import sys
import platform

OPERATING_SYSTEM = platform.system()  # 현재 해당 파일이 실행되고 있는 운영체제
MAIN_BASE_PATH = ""

if OPERATING_SYSTEM == "Darwin":
    MAIN_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
elif OPERATING_SYSTEM == "Linux":
    MAIN_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
elif OPERATING_SYSTEM == "Windows":
    if hasattr(sys, "_MEIPASS"):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        MAIN_BASE_PATH = os.path.join(sys._MEIPASS, "config")
    else:
        MAIN_BASE_PATH = os.path.dirname(os.path.abspath(__file__))
