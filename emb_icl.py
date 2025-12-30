from pyspark.sql import SparkSession
import logging
import time
import cv2
from pyspark.sql.types import *
import json
import numpy as np
import pickle
import requests
from time import sleep
import requests
from concurrent.futures import ThreadPoolExecutor
from viking.vikingdb_client import VikingDbData, VikingDbClient
import euler
euler.install_thrift_import_hook()
import os
from pyspark.sql import SparkSession
spark = SparkSession \
        .builder \
        .enableHiveSupport() \
        .config('spark.speculation', True) \
        .config('spark.sql.adaptive.maxNumPostShufflePartitions', 10000) \
        .config('spark.shuffle.hdfs.enabled', True) \
        .config('spark.shuffle.io.maxRetries', 1) \
        .config('spark.shuffle.io.retryWait', '0s') \
        .config('spark.network.timeout', '300s') \
        .config('spark.merge.files.enabled', True) \
        .config('spark.merge.files.number', 10) \
        .getOrCreate()

AK = '609bd0d8fa5e5b11104b3ab2160386ba'
SK = 'b9869d24618d3a483b9d9b1a498294bc'
TARGET_LABEL = '诱导互动'
AUDIT_TYPE = 'MCQ'

PROJECT_ID = '诱导互动合并版_20250523_caption_v4'
LABEL_UPPER64 = 6

# PROJECT_ID = '诱导互动合并版_20250513_caption_thinking_v2'
# LABEL_UPPER64 = 3

white_img = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xdb\x00C\x01\t\t\t\x0c\x0b\x0c\x18\r\r\x182!\x1c!22222222222222222222222222222222222222222222222222\xff\xc0\x00\x11\x08\x01\x00\x01\x00\x03\x01\"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13\"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a&'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xf9\xfe\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x0f\xff\xd9"

require_frames_num = 16
def get_img_content_type_num(info_item_dicts,item,get_frame_func):
    getMLFramesByNum = get_frame_func
    tmp_ret = []
    item_type =  0 if( 'video_id' in info_item_dicts[item].Content) else 1 
    cut_times = [0] * require_frames_num
    # item_media_type = '4' 视频
    if( 'video_id' in info_item_dicts[item].Content):
        frames = getMLFramesByNum(info_item_dicts[item].Content['video_id'],AK,SK,FrameNum=require_frames_num,NeedBinaryInfo=False,Caller="aweme.account.user_originality_offline").Frames
        frames_urls = [f.DownloadURL for f in frames]
        cut_times = [f.CutTime for f in frames]

        if(len(frames_urls) <= 0): 
            print(item,len(frames_urls))
        
        tmp_ret = download_img_url_list(frames_urls)
    # item_media_type = '42' slides
    elif 'clips' in info_item_dicts[item].Content:
        clips = info_item_dicts[item].Content['clips']
        true = True
        false = False
        clips = eval(clips)
        for clip in clips:
            clip_type = clip['clip_type']
            if clip_type == 4:
                video_id, duration = clip.get('video_id', ''), clip.get('duration', 0)
                frames = getMLFramesByNum(video_id,AK,SK,FrameNum=require_frames_num,NeedBinaryInfo=False,Caller="data.content.multimodal").Frames
                frames_urls = [f.DownloadURL for f in frames]
                if(len(frames_urls) <= 0): 
                    print(item,len(frames_urls))
                
                clip_ret = download_img_url_list(frames_urls)
                tmp_ret.extend(clip_ret)
            elif clip_type == 2:
                uri_prefix = 'https://p-image-tns-cs.bytedance.net/obj/'
                url = uri_prefix + clip['uri']
                raw_data = download_img_url_list([url])[0]
                tmp_ret.append(raw_data)
            else:
                print("uknown type")

    # item_media_type = '2' 图集，部分slides 会有'images'的字段
    elif('images' in info_item_dicts[item].Content):
        uri_prefix = 'https://p-image-tns-cs.bytedance.net/obj/'
        images = info_item_dicts[item].Content['images']
        urls = [ uri_prefix + m['uri'] for m in images][:require_frames_num]
        
        if(len(urls) <= 0): 
            print(item,len(urls))
        
        tmp_ret = download_img_url_list(urls)
    
    item_num = len(tmp_ret)
    # while(len(tmp_ret) < require_frames_num):
    #     tmp_ret.append(white_img)
    return (tmp_ret,item_type,item_num,cut_times)

from pyspark.sql.types import *

def cleanDescription(desc):
    splits = desc.split(';')
    return ';'.join([s for s in splits if len(s) > 1])

def get_ocr_text(attr_item_dicts,item_id):
    detail_video_ocr =  json.loads(attr_item_dicts[item_id].ItemAttrMap.get('aweme_ocr','{"integrated_text":""}'))['integrated_text']
    detail_video_sentence_ocr = attr_item_dicts[item_id].ItemAttrMap.get('ai_ocr_sentence','')
    ocr_text = attr_item_dicts[item_id].ItemAttrMap.get("ocr_text",'')
    if len(detail_video_ocr) >= 2: return detail_video_ocr
    elif len(ocr_text) >=2 : return ocr_text
    else: return detail_video_sentence_ocr

def retry_download_url(url,max_try = 2, timeout = 1.5):
    while(max_try):
        try:
            img_content = requests.get(url, stream=False, timeout = timeout).content
            assert(len(img_content) > 100)
            return img_content
        except:
            max_try -= 1
    return white_img

def download_img_url_list(urls,timeout = 10):
    res = []
    with ThreadPoolExecutor(max_workers=max(len(urls),1)) as e:
        res = e.map(retry_download_url, urls)
    return list(res)

def get_frames(frame_urls):
    if frame_urls is None or frame_urls == '': return None
    if frame_urls[0] != '[' or frame_urls[-1] != ']': return None
    frame_list = eval(frame_urls)
    urls = []
    for frame_info in frame_list:
        url = frame_info.get('image_url', '')
        if url is None or url == '': continue
        urls.append(url)
    return download_img_url_list(urls)

def get_frames_by_item_id(iteminfo_func, frames_func, item_id):
    info_item_dicts = None
    max_try = 2
    while(max_try>0):
        max_try -= 1
        info_item_dicts = iteminfo_func([item_id])
        if len(info_item_dicts):
            break
    if(max_try <= 0):
        return None

    f1,t1,n1,cut_times = get_img_content_type_num(info_item_dicts, item_id, frames_func)
    return f1

cluster = "abase_content_gandalf_aweme"
abase_table = "aweme_audio_profile"
import json

import bytedabase
class AsrAbaseProvider:
    def __init__(
        self,
        db_name="abase_content_gandalf_aweme",
        table_name="aweme_audio_profile",
        text_key="audio_asr_text",
        item_type="music",  # ["music", "item"]
    ):
        self.abase_client = bytedabase.Client(psm=db_name, table=table_name)
        self.table_name = table_name
        self.text_key = text_key
        self.item_type = item_type

    def fetch(self, object_id, retry=1):
        """
        object_id: item_type是music的情况下，传music id。 item的情况下，传视频id。
        """
        for i in range(retry):
            try:
                key = f"[{self.table_name}]general_abase_aweme_audio_info_{self.item_type}_{object_id}"
                ret = self.abase_client._redis_client.get(key)
                if ret is None:
                    return {}
                texts = json.loads(ret)
                return texts
            except Exception as e:
                logging.error("get_from_abase with key:{} failed: {}".format(key, e))
                continue
        return {}

def get_music_id_asr(music_id,music_client):
    key = f"[aweme_audio_profile]general_abase_aweme_audio_info_music_{music_id}"
    ret = music_client.abase_client._redis_client.get(key)
    if(ret is None): return ''
    js_obj = json.loads(ret)
    ret = js_obj.get('audio_asr_text','')
    return ret

def get_asr_text(info_item_dicts,attr_item_dicts,item_id,music_client,music_abase_provider):
    music_id = info_item_dicts[item_id].Content['music'] if 'music' in info_item_dicts[item_id].Content else ''
    detail_music = get_music_id_asr(music_id,music_client)
    ai_asr = attr_item_dicts[item_id].ItemAttrMap.get('ai_asr','')
    return (detail_music if len(detail_music) else ai_asr),music_abase_provider.fetch(str(music_id))

def get_cover_ocr(info_item_dicts,item_id, coverocr_func):
    uri_prefix = 'https://p-image-tns-cs.bytedance.net/obj/'
    cover_url = info_item_dicts[item_id].Content.get("cover_image_uri", "")
    if cover_url:
        cover_url = uri_prefix + cover_url
        cover_data = download_img_url_list([cover_url])[0]
        cover_ocr = coverocr_func([item_id])[item_id]["item_cover_ocrtext"]
        if cover_ocr == "tmp": 
            cover_ocr = ''
        return cover_data, cover_ocr
    return None,None

def get_ocr_asr_title_from_item_id(item_id, iteminfo_func, itemattr_func, music_client, asr_abase_provider):
    attr_item_dicts = None
    info_item_dicts = None
    max_try = 2
    while(max_try>0):
        max_try -= 1
        attr_item_dicts = itemattr_func([item_id])
        info_item_dicts = iteminfo_func([item_id])
        if(len(attr_item_dicts) and len(info_item_dicts)):
            break
        time.sleep(1)

    if(max_try <= 0):
        return None

    asr,music_feature = get_asr_text(info_item_dicts,attr_item_dicts,item_id,music_client,asr_abase_provider)
    ocr = get_ocr_text(attr_item_dicts,item_id)
    try:
        title = info_item_dicts[item_id].Content['text']
    except:
        title = ''
    return {'title':title, 'ocr':ocr, 'asr':asr}

def safe_json_loads(raw_str):
    try:
        data = json.loads(raw_str)
    except Exception:
        data = {}
    return data

##################################### Call Uniclipv4 Server ###############################

def uniclipv4Infer(one_case, uniclipv4_client):
    from shopkeeper.thrift_retrievers.idl import fermion_thrift
    from shopkeeper.thrift_retrievers.idl.fermion_thrift import Tensor, DataType
    import numpy as np
    import base64
    object_id = one_case['object_id']
    limit_frames = [bytes(x) for x in one_case['frames']]
    frame_len = len(limit_frames)
    if frame_len == 0:
        return None
    elif frame_len <=8:
        while len(limit_frames)<8:
            limit_frames.append(limit_frames[-1])
    else:
        limit_frames = limit_frames[::frame_len//8]
        if len(limit_frames) > 8: limit_frames = limit_frames[:8]
    # frame
    frames_data = Tensor()
    frames_data.dtype = DataType.STRING
    frames_data.str_data = limit_frames
    frames_data.shape = [8]
    title = one_case['title']
    ocr = one_case['ocr']
    asr = one_case['asr']
    if title is None: title = ""
    if ocr is None: ocr = ""
    if asr is None: asr = ""
    # title_text
    title_data = Tensor()
    title_data.dtype = DataType.STRING
    title_data.str_data = [title[:64]]
    title_data.shape = []
    # ocr_text
    ocr_data = Tensor()
    ocr_data.dtype = DataType.STRING
    ocr_data.str_data = [ocr[:128]]
    ocr_data.shape = []
    # asr_text
    asr_data = Tensor()
    asr_data.dtype = DataType.STRING
    asr_data.str_data = [asr[:256]]
    asr_data.shape = []
    req = fermion_thrift.InferRequest()
    tensors = {
        "image": frames_data,
        "pure_title": title_data,
        "ocr_merge_text": ocr_data,
        "asr_background": asr_data
    }
    req.input = [fermion_thrift.TensorSet(tensors=tensors)]
    max_retry_num = 3
    try_id = 0
    try_success = False
    while try_id < max_retry_num and try_success == False:
        try:
            sleep(1)
            uniclipv4_resp=uniclipv4_client.Infer(req)
            uniclipv4_embedding = uniclipv4_resp.output[0].tensors['embedding0'].float_data # dim = 128 
            # uniclipv4_embedding = uniclipv4_resp.output[0].tensors['embedding1'].float_data + \
            #             uniclipv4_resp.output[0].tensors['embedding2'].float_data + \
            #             uniclipv4_resp.output[0].tensors['embedding3'].float_data # dim=2304
            # uniclipv4_embedding=np.array(uniclipv4_embedding)
            one_case['uniclipv4_embedding'] = uniclipv4_embedding #base64.b64encode(uniclipv4_embedding.astype(dtype=np.float16).tobytes()).decode("ascii")
            one_case.pop('frames_bytearray_list', None)
            try_success = True
        except Exception as e:
            print(e,'try_id:{}, uniclipv4_embedding {} is None !'.format(try_id, one_case['object_id']))
            try_success = False
            one_case['uniclipv4_embedding'] = None
        try_id += 1
    if try_success == False:
        one_case['uniclipv4_embedding'] = None
    return one_case['uniclipv4_embedding']

##################################### Read Universal Embedding #############################
def make_fid_v1(value, slot):
    import cityhash 

    value_mask_v1 = (1 << 54) - 1

    if isinstance(value, str):
        return (cityhash.CityHash64(value) & value_mask_v1) | (slot << 54)
    elif isinstance(value, int):
        return (value & value_mask_v1) | (slot << 54)
    else:
        raise TypeError("value must be type in [str, int] in make_fid_v1")

def convert_fid_v1_to_v2(fid):
    slot = fid >> 54
    return (slot << 48) | (fid & 0x0000FFFFFFFFFFFF)

def make_fid_v2(value, slot):
    return convert_fid_v1_to_v2(make_fid_v1(value, slot))

def get_ue_emb_by_fid(
    uid_list, embedding_source_name="fc_aweme_sec_graphsage_unsup_r2186633_d32_0", slot=1
):
    import euler
    from shopkeeper.thrift_retrievers.idl.base_thrift import Base
    from shopkeeper.thrift_retrievers.service_rpc_idl.lagrange.universal_embedding.universal_embedding_thrift import (
        UniversalEmbeddingService,
        Request,
        Response,
        MultiRequest,
        MultiResponse,
        SliceConfig,
        EmbeddingSource,
        EmbeddingInfo,
        UIDInfo,
        IDType,
        FIDInfo,
    )
    """
    对于用户GE请求，传入int id， 对于投稿CLIP请求，传入str id
    """
    ue_client = euler.Client(UniversalEmbeddingService, "sd://data.embedding.feed?cluster=default", timeout=10)
    fid_list = [make_fid_v2(uid, slot=slot) for uid in uid_list]
    list_fid_infos = [FIDInfo(fids=[fid]) for fid in fid_list]

    source = EmbeddingSource(embedding_name=embedding_source_name)
    ebd = EmbeddingInfo(source=source)
    req = Request(
        fid_infos=list_fid_infos,
        embedding_info=ebd,
        Base=Base(Caller="aweme.account.feature_proxy"),
    )

    res = ue_client.get_embedding(req)

    return res

##################### Write VikingDB #######################
def get_embed(item_info, uniclipv4_client):
    """根据item_id获取embed
    """
    item_id = str(item_info['object_id'])
    #抽取video+txt特征
    video_txt_feature = get_ue_emb_by_fid([item_id], 'fc_aweme_first_uniclip4_mm_emb', slot=2).fid_rsps[0].val
    if sum(video_txt_feature) < 1e-8:
        fvector = uniclipv4Infer(item_info, uniclipv4_client)
        if fvector is not None:
            fvector = (fvector / np.linalg.norm(fvector, ord=2)).tolist()#归一化
    else:
        fvector = video_txt_feature
        fvector = (fvector / np.linalg.norm(fvector, ord=2)).tolist()#归一化
    return fvector

import numpy as np
import datetime
import json
from collections import Counter

def get_weights(time_day):
    """7天内2，7-14天1.5， 14-21天1， 21天以后0.5
    """
    if time_day < 7:
        return 2
    elif time_day < 14:
        return 1.5
    elif time_day < 21:
        return 1.0
    else:
        return 0.5

def policy_title_filter(policy_title):
    if policy_title is None or policy_title == '' or policy_title == 'None': return ''
    policy_title = policy_title.strip()
    if 'bpo' in policy_title:
        policy_title = '-'.join(policy_title.split('-')[1:-1])
    if '【专审】' in policy_title:
        policy_title = policy_title.split('】')[1].strip()
    if '-A' in policy_title:
        policy_title = policy_title.replace('-A','')
    return policy_title

def rerank(item_id, candidates, filter_num = 5, min_score = 0.01):
    import random
    if candidates == []: return None
    scores = [] 
    item_infos = [] 
    dates = []

    already_add_item_ids = []
    for item in candidates:
        cand_item_id = item['label_lower64']
        cand_score = item['scores']
        if(cand_item_id not in already_add_item_ids):
            already_add_item_ids.append(cand_item_id)
        else:
            continue
        
        # if cand_item_id == item_id or cand_score < min_score: continue
        
        if cand_item_id == item_id: 
            continue
        
        cand_attrs = json.loads(item['attrs'])
        
        # cand_policy_title = policy_title_filter(str(cand_attrs["policy_title"]))
        # if cand_policy_title is None or cand_policy_title == 'None' or cand_policy_title == '': continue

        cand_date = cand_attrs["date"]

        scores.append(cand_score)
        dates.append(cand_date)
        
        item_infos.append({
            "item_id": item['label_lower64'],
            "title": cand_attrs["title"],
            "ocr": cand_attrs["ocr"],
            "asr": cand_attrs["asr"],
            "important_text": json.loads(cand_attrs.get('important_text_dict','{}')).get(TARGET_LABEL,''),
            "attrs": cand_attrs,
            "sim_scores": cand_score
        })
        
    return item_infos[:filter_num]

RISK_DOMAINS = '诱导互动'

POS_DSL = { "op": "and",
            "conds": [{
            "op": "must",
            "field": "project_id",
            "conds": [PROJECT_ID]
            },
        {
        "op": "or", 
        "conds":[
            {
                "op": "must",           
                "field": "selected_title_0",      
                "conds": [TARGET_LABEL]   
            },
            {
                "op": "must",           
                "field": "selected_title_1",      
                "conds": [TARGET_LABEL]   
            },
            {
                "op": "must",           
                "field": "selected_title_2",      
                "conds": [TARGET_LABEL]   
            },
            {
                "op": "must",           
                "field": "selected_title_3",      
                "conds": [TARGET_LABEL]   
            },
            {
                "op": "must",           
                "field": "selected_title_4",      
                "conds": [TARGET_LABEL]   
            },
        ]
    }]
}

NEG_DSL = {
        "op": "and", 
        "conds":[
            {
                "op": "must_not",           
                "field": "selected_title_0",      
                "conds": [TARGET_LABEL]   
            },
            {
                "op": "must_not",           
                "field": "selected_title_1",      
                "conds": [TARGET_LABEL]   
            },
            {
                "op": "must_not",           
                "field": "selected_title_2",      
                "conds": [TARGET_LABEL]   
            },
            {
                "op": "must_not",           
                "field": "selected_title_3",      
                "conds": [TARGET_LABEL]   
            },
            {
                "op": "must_not",           
                "field": "selected_title_4",      
                "conds": [TARGET_LABEL]   
            },
            {
                "op": "must",           
                "field": "project_id",      
                "conds": [PROJECT_ID]
            }
        ]
    }

def get_dsl(label_dsl, risk_domins):
    return {
        "filter" :{
        "op": "and", 
        "conds":[ label_dsl, {
                "op": "must",           
                "field": "risk_domains",      
                "conds": [risk_domins]   
            } ]
        }
        }
    
    

def process_row(row, vikingdb_client, uniclip_client,  iteminfo_func, frames_func):
    
    row_dict = row.asDict()

    row_dict["object_id"] = row_dict["item_id"]
    item_id = row_dict["item_id"]
    selected_title = row_dict["selected_title"]
    video_id = row_dict["video_id"]
    item_title = row_dict["item_title"]
    ocr_text = row_dict["ocr_text"]
    asr_text = row_dict["asr_text"]
    frames = row_dict["frames"]
    date = '${date}'
    
    row_dict['title'] = row_dict['item_title']
    row_dict['ocr'] = row_dict['ocr_text']
    row_dict['asr'] = row_dict['asr_text']
    # row_dict['frames'] = row_dict['frames']
    item_media_type = str(iteminfo_func([item_id])[item_id].MediaType)
    
    # try to read from vikingdb
    # ret_info = vikingdbget_data_client.({"label_lower64": int(item_id), "label_upper64": LABEL_UPPER64})
    # is_found = ret_info[0]
    # if is_found:
    #     fvector = ret_info[1][0]['fvector']
    # else:
    
    fvector = get_embed(row_dict, uniclip_client)
    
    if fvector is None: 
        print(item_id,item_title,ocr_text,asr_text,len(frames),' is failed to get embed.')
        return None

    # basic info
    asr = row_dict["asr_text"]
    ocr = row_dict["ocr_text"]
    title = row_dict["item_title"]

    MAX_TTL_DAY = 90
    MODEL_NAME = "Uniclipv4"
    MODALITY = "imagetext"
    version = 'V0'

    topK = 5
    TTL = 90 # days
    date_now = datetime.datetime.strptime('${date}', '%Y%m%d')
    query_dates = [(date_now - datetime.timedelta(days=delta)).strftime('%Y%m%d') for delta in range(0, TTL+1)]
    retrieve_topK = 20

    # rec_policies v1 -> query all policy_title of the label_1 domains in the specified project_id
    
    try:
        ret_info = vikingdb_client.recall(fvector, index=version, topk=retrieve_topK, dsl_query= get_dsl(POS_DSL, RISK_DOMAINS) ) 
        status, cands = ret_info[0], ret_info[1]
    except:
        rec_policies_v1 = []
        cands = []
          
    if status == False:
        rec_policies_v1 = []
        cands = []
    
    pos_rerank_result = rerank(int(item_id), cands, topK)
    # rec_policies v2 -> query all policy_title of project_id
    try:
        ret_info = vikingdb_client.recall(fvector, index=version, topk=retrieve_topK, dsl_query = get_dsl(NEG_DSL, RISK_DOMAINS) ) 
        status, cands = ret_info[0], ret_info[1]
    except:
        # return None
        rec_policies_v2 = []
        cands = []
    if status == False:
        rec_policies_v2 = []
        cands = []

    neg_rerank_result = rerank(int(item_id), cands, topK)

    if(neg_rerank_result is None):
        print(item_id,' is failed to recall icl examples.')
        return 
    pos_rerank_result = pos_rerank_result if pos_rerank_result is not None else []
    
    # "item_id": item['label_lower64'],
    # "title": cand_attrs["title"],
    # "ocr": cand_attrs["ocr"],
    # "asr": cand_attrs["asr"],
    # "attrs": cand_attrs,
    # "sim_scores": scores

    for item_info in pos_rerank_result:
        item_id = int(item_info['item_id'])
        frames = get_frames_by_item_id(iteminfo_func, frames_func, item_id)
        item_info['frames'] = frames
    
    for item_info in neg_rerank_result:
        item_id = int(item_info['item_id'])
        frames = get_frames_by_item_id(iteminfo_func, frames_func, item_id)
        item_info['frames'] = frames
        
    pos_rerank_result = pickle.dumps(pos_rerank_result)
    neg_rerank_result = pickle.dumps(neg_rerank_result)
    
    return row + (pos_rerank_result,neg_rerank_result)

def mfunc(rows):
    TOKEN = "a7c9f2469b3f70d50672bf89cfebf923"
    VIKINGDB_NAME= "contentsecurity_recall_1746512370__eco_audit_video_recall_online"
    #初始化viking
    vikingdb_client = VikingDbClient(vikingdb_name=VIKINGDB_NAME, token=TOKEN, region="CN") #CN BOE VA SG GCP
    MAX_TTL_DAY = 90 #最多存放时间
    
    #########################  Uniclip v4 Client ###################################
    from shopkeeper.thrift_retrievers.idl.fermion_core_thrift import FermionCore
    uniclipv4_client = euler.Client(FermionCore, 'sd://data.content.uniclip_v4_emb_offline_for_stamp?cluster=default&idc=hl', timeout=40)

    from shopkeeper import getItemInfo2Dict, getMLFramesByNum
    
    ret = []
    for idx, row in enumerate(rows):
        tmp = process_row(row, vikingdb_client, uniclipv4_client, getItemInfo2Dict, getMLFramesByNum)
        if(tmp != None):
            ret.append(tmp)
        if idx == 0:
            print('Processing Object',row)
            print("Result:\t", tmp)
        # sleep(0.2)
    return ret

######################. Main Function #######################

########### VikingDB ################

####################### DataFrame ############################
from pyspark.sql import functions as F
from pyspark.sql.types import *
from functools import partial

# read_data_path = 'hdfs://haruna/home/byte_content_security/user/liuzejun.1219/vlm/bmk/induce_interaction/mcq/${date}'
# output_path = 'hdfs://haruna/home/byte_content_security/user/liuzejun.1219/vlm/bmk/induce_interaction/icl/' + 'with_important_text' +  '/${date}'

# hdfs://haruna/home/byte_content_security/user/liuzejun.1219/vlm/bmk/induce_interaction/mcq/vvr

read_data_path = 'hdfs://haruna/home/byte_content_security/user/caizerui/induce_interaction/${date}_with_caption_important_text_analysis_qa'
output_path = 'hdfs://haruna/home/byte_content_security/user/caizerui/induce_interaction/${date}_with_caption_important_text_analysis_qa_with_icl_v4' 

input_data_df = spark.read.parquet(read_data_path)

original_schema = input_data_df.schema
# 创建新的 schema，添加列

# pos_rerank_result,neg_rerank_result
schema = StructType(original_schema.fields + [
    StructField("pos_icl_cands", BinaryType(), True),
    StructField("neg_icl_cands", BinaryType(), True)
])

output_data = input_data_df.rdd.repartition(100).mapPartitions(mfunc)
output_df = spark.createDataFrame(output_data,schema)
output_df.write.parquet(output_path,mode = 'overwrite')
