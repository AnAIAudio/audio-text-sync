import subprocess


def run(
    data_path: str,
    dict_path: str,
    acoustic_path: str,
    output_path: str,
    config_path: str,
):
    try:
        result = subprocess.run(
            [
                "mfa",
                "align",
                data_path,
                dict_path,
                acoustic_path,
                output_path,
                "--config_path",
                config_path,
                "--output_format",
                "json",
                "--num_jobs",
                "12",
                "--clean",
            ],
            check=True,
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode

def merge(aligned_texts: dict, text_list: list):
    merged_segments = []
    idx = 0
    max_length = len(aligned_texts["tiers"]["words"]["entries"])

    for text in text_list:
        segment = {"start": int(1e9), "end": -1, "text": text}
        text = text.lower()

        while idx < max_length:
            if aligned_texts['tiers']['words']['entries'][idx][2] == "[bracketed]":
                idx += 1
                continue

            if aligned_texts["tiers"]["words"]["entries"][idx][2].lower() not in text:
                break

            text = text.replace(aligned_texts["tiers"]["words"]["entries"][idx][2].lower(), "", 1)
            segment["start"] = min(segment["start"], aligned_texts["tiers"]["words"]["entries"][idx][0])
            segment["end"] = max(segment["end"], aligned_texts["tiers"]["words"]["entries"][idx][1])
            idx += 1

        if segment["text"] and segment['start'] != int(1e9) and segment['end'] != -1:
            merged_segments.append(segment)

    return merged_segments