from google.colab import auth
from oauth2client.client import GoogleCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class googleDrive():
    def __init__(self):
        self.drive = self.login_google_drive()

    def login_google_drive(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)
        return drive

    def list_file(self, folder='root'):
        command = '\''+folder+'\''+' in parents and trashed=false'
        file_list = self.drive.ListFile({'q': command}).GetList()
        for files in file_list:
            print('title: %s, id: %s, mimeType: %s' %
                  (files['title'], files['id'], files["mimeType"]))

    def downloadFile(self, id, name):
        self.drive.CreateFile({'id': id}).GetContentFile(name)

    def uploadFile(self, name):
        upload = self, drive.CreateFile({'title': name})
        upload.SetContentFile(name)
        upload.Upload()
