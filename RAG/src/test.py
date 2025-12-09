from http import HTTPStatus
import dashscope
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
dashscope.api_key='sk-cdcfd10a38c74e63a7b79ebc4234256d'
messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '如何做西红柿鸡蛋？'}]

response = dashscope.Generation.call(
    model='qwen-plus',
    messages=messages,
    result_format='message',  # set the result to be "message" format.
)

if response.status_code == HTTPStatus.OK:
    print(response)
else:
    print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
        response.request_id, response.status_code,
        response.code, response.message
    ))