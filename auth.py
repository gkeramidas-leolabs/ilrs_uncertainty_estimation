import os

api_url = os.environ.get("LEO_API_URL")
leo_labs_user = os.environ.get("LL_ACCESS_KEY")
leo_labs_key = os.environ.get("LL_SECRET_KEY")
headers = {"Authorization": "".join(["basic ", leo_labs_user, ":", leo_labs_key])}
