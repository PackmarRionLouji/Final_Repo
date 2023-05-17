import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account


CLIENT_SECRET_FILE = 'credentials.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCOPES = ['https://www.googleapis.com/auth/drive']

def create_service():
    credentials = service_account.Credentials.from_service_account_file(CLIENT_SECRET_FILE, scopes=SCOPES)
    service = build(API_NAME, API_VERSION, credentials=credentials)
    return service

def download_file(file_id, file_name):
    service = create_service()
    request = service.files().get_media(fileId=file_id)

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print('Download progress: {}%'.format(int(status.progress() * 100)))
    
    fh.seek(0)
    with open(file_name, 'wb') as f:
        f.write(fh.read())
    
    print('File downloaded successfully.')

file_id = '1rs8POlleS_nGKBIaZ6lb6-ZwbonNaWVh'
file_name = 'pneumonia_detection_model (1).h5'

download_file(file_id, file_name)
