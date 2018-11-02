import time
import os
from zipfile import ZipFile
folder = r''
zip_file_name = r'submissions\{}_submission.zip'.format( time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) )
    
files = [f for f in os.listdir(path='.') if f.endswith('.py')]
files.append('metadata.json')
files.append('holidays.csv')


with ZipFile(zip_file_name, mode='w') as submission:
    for file in files:
        submission.write(folder + file,arcname=file)
submission.close()    

print('Summission {} prepared'.format(zip_file_name))