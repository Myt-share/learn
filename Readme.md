# 文字转语音

> **支持音色选择**  
> 女音色：`zf_001 ~ zf_099`  
> 男音色：`zm_001 ~ zm_100`

```bash
curl -X POST 'http://127.0.0.1:20240/tts' \
  -H 'accept: */*' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "您好，一般情况下,购电卡在装表或换表时由我公司工作人员现场发放；少量换表客户需到营业厅激活，客户需要提供新装表单或者换表凭证，第一次领取购电卡，不收取费用。",
    "voice": "zf_072"
  }' \
  -o output.mp3
```

#  语音转文字

```bash
curl -X 'POST' \
  'http://127.0.0.1:20240/asr' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@范例.wav;type=audio/wav'
```
