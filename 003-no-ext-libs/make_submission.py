import time
from zipfile import ZipFile
folder = r''
zip_file_name = folder + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '_submission.zip'
    
files = [
    'metadata.json',
    'predict.py',
    'train.py',
    'utils.py']

with ZipFile(zip_file_name, mode='w') as submission:
    for file in files:
        submission.write(folder + file,arcname=file)
submission.close()    

print('Summission {} prepared'.format(zip_file_name))