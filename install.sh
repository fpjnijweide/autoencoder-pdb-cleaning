python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout jupyter/mykey.key -out jupyter/mycert.pem
deactivate