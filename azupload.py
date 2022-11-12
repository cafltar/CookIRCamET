
from azure.identity import DefaultAzureCredential,ClientSecretCredential
from azure.storage.blob import BlobServiceClient
import os

def upload_file(client,source, dest):
    '''
    Upload a single file to a path inside the container
    '''
    print(f'Uploading {source} to {dest}')
    with open(source, 'rb') as data:
        client.upload_blob(name=dest, data=data, overwrite=True)
        
def upload_dir(client, source, dest):
    '''
    Upload a directory to a path inside the container
    '''
    prefix = '' if dest == '' else dest + '/'
    prefix += os.path.basename(source) + '/'
    for root, dirs, files in os.walk(source):
        for name in files:
            dir_part = os.path.relpath(root, source)
            dir_part = '' if dir_part == '.' else dir_part + '/'
            file_path = os.path.join(root, name)
            blob_path = prefix + dir_part + name
            upload_file(client,file_path, blob_path)
# In[14]:


f=open('.netrc','r')

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


work = blob_service_client.get_container_client("work")


upload_dir(raw,'../../raw/CookIRCamET/','CookIRCamET')
upload_dir(work,'../../work/CookIRCamET/','CookIRCamET')

