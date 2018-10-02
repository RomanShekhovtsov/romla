import time
from zipfile import ZipFile
folder = r''
zip_file_name = r'submissions\{}_submission.zip'.format( time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) )
    
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