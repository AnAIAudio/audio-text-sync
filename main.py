if __name__ == "__main__":
    import os
    from custom_path import MAIN_BASE_PATH
    from scb_dtw import run_dtw

    TEMP_DIRECTORY_PATH = os.path.join(MAIN_BASE_PATH, 'temp')
    AUDIO_DIRECTORY_PATH = os.path.join(MAIN_BASE_PATH, 'audio')
    TEXT_DIRECTORY_PATH = os.path.join(MAIN_BASE_PATH, 'text')

    if not os.path.exists(TEMP_DIRECTORY_PATH):
        os.makedirs(TEMP_DIRECTORY_PATH, exist_ok=True)

    if not os.path.exists(AUDIO_DIRECTORY_PATH):
        os.makedirs(AUDIO_DIRECTORY_PATH, exist_ok=True)

    if not os.path.exists(TEXT_DIRECTORY_PATH):
        os.makedirs(TEXT_DIRECTORY_PATH, exist_ok=True)

    run_dtw()
