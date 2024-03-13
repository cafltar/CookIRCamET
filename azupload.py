from azure.identity import DefaultAzureCredential,ClientSecretCredential
from azure.storage.blob import BlobServiceClient
import os
from datetime import datetime
import sys
import pytz
def upload_file(client,source, dest):
    '''
    Upload a single file to a path inside the container
    '''
    with open(source, 'rb') as data:
        try:
            print(f'Uploading {source} to {dest}')
            client.upload_blob(name=dest, data=data, overwrite=False)
        except Exception as e: 
            print(e)
            print('Exists')
            #check last modified date
            source_date = datetime.fromtimestamp(os.path.getmtime(source), tz=pytz.UTC)
            b = [b for b in work.list_blobs(dest)]
            if len(b)>0:
                dest_date = b[0].last_modified
                if source_date>dest_date:
                    print(f'Newer verion: uploading {source} to {dest}')
                    client.upload_blob(name=dest, data=data, overwrite=True)


f=open('.azure','r')

for l in f:    
    if l.split('=')[0]=='tenant':
        tenant=l.split('=')[1][:-1]
    if l.split('=')[0]=='appId':
        app=l.split('=')[1][:-1]
    if l.split('=')[0]=='Secret_ID':
        client=l.split('=')[1][:-1]
    if l.split('=')[0]=='Secret_Value':
        secret=l.split('=')[1][:-1]
    if l.split('=')[0]=='endpoint':
        url=l.split('=')[1][:-1]
f.close()

credential = ClientSecretCredential(tenant_id=tenant, client_id=app, client_secret=secret)
blob_service_client = BlobServiceClient(account_url=url, credential=credential)

raw = blob_service_client.get_container_client("raw")
print(raw)
