incase whisper poops the bed run this (replace model)

- Test the SSL bypass with manual download
```
python3 -c "
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import whisper
whisper.load_model('small')
"
```