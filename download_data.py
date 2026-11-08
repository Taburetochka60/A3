"""Download the official Ukrainian table in resumable HTTP byte ranges via Kaggle auth."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests,time,threading,argparse
local=threading.local()
from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDataFileRequest
BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--file', default='uk_geo_dataset_processed_v1.parquet', choices=['uk_geo_dataset.csv','uk_geo_dataset_processed_v1.parquet'])
args=parser.parse_args()
api=KaggleApi();api.authenticate()
for attempt in range(5):
 try:
  with api.build_kaggle_client() as client:
   req=ApiDownloadDataFileRequest();req.competition_name='kse-ua-location-extraction-2026';req.file_name=args.file
   response=client.competitions.competition_api_client.download_data_file(req)
   url=response.request.url
  probe=requests.get(url,headers={'Range':'bytes=0-0'},timeout=30)
  assert probe.status_code==206
  size=int(probe.headers['Content-Range'].split('/')[-1]);url=probe.url;break
 except Exception:
  if attempt==4:raise
parts=BASE/'data'/(args.file.replace('.','_')+'_parts_64k');parts.mkdir(exist_ok=True)
chunk=64*1024
n=(size+chunk-1)//chunk

def fetch(i):
 start=i*chunk;end=min(size,start+chunk)-1
 dest=parts/f'{i:04d}'
 if dest.exists() and dest.stat().st_size==end-start+1:return i
 for attempt in range(6):
  try:
   if not hasattr(local, 'session'):local.session=requests.Session()
   r=local.session.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(15,90))
   expected=f'bytes {start}-{end}/{size}'
   if r.status_code!=206 or len(r.content)!=end-start+1 or r.headers.get('Content-Range')!=expected:
    raise RuntimeError('Invalid range response')
   dest.write_bytes(r.content);return i
  except Exception:
   if attempt==5:raise RuntimeError(f'Part {i} failed') from None
   time.sleep(1)
with ThreadPoolExecutor(max_workers=48) as pool:
 for done,f in enumerate(as_completed([pool.submit(fetch,i) for i in range(n)]),1):
  f.result()
  if done%100==0 or done==n:print(f'{done}/{n} parts',flush=True)
dest=BASE/'data'/args.file
with dest.open('wb') as out:
 for i in range(n):out.write((parts/f'{i:04d}').read_bytes())
assert dest.stat().st_size==size
print('Download complete',size,flush=True)

