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
