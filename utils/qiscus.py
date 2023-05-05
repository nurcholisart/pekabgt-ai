import requests


def upload_file(app_code: str, file_path: str) -> str:
    headers = {"QISCUS_SDK_APP_ID": app_code}

    resp = requests.post(
        "https://api3.qiscus.com/api/v2/sdk/upload", headers=headers, files={"file": open(file_path, "rb")}
    )

    json_resp = resp.json()
    url = json_resp["results"]["file"]["url"]

    return url
