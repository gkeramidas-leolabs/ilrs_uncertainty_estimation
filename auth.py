import os

api_url = os.environ.get('LEO_API_URL')
leo_labs_user = os.environ.get('Access_Key')
leo_labs_key = os.environ.get('Secret_Key')
headers = {'Authorization': ''.join(['basic ', leo_labs_user, ':', leo_labs_key])}