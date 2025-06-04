import requests
import json
# Flask应用的URL
url = 'http://127.0.0.1:5000'

def getAnswer(question:str,model:str='basicModel')->str:
    posturl=url+f'/{model}/{question}'
    try:
        response = requests.post(posturl)
        if response.status_code == 200:
            result = json.loads(response.text)
            return result['answer']
        else:
            return ''
    except:
        return ''
if __name__ == '__main__':
    print(getAnswer('你好'))
