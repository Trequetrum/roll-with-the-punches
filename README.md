#Roll with the Punches


### If your OS Maintains compatibility with the Ubuntu repositories

With your terminal's current working directory (CWD) in the root folder of this project, the following commands should set up and run this project

- `sudo apt update && sudo apt upgrade -y`
- `sudo add-apt-repository ppa:deadsnakes/ppa`
- `sudo apt install python3.8`
- `sudo apt install python3.8-venv`
- `python3.8 -m venv .venv`
- `source .venv/bin/activate`
- `python -m pip install --upgrade pip`
- `python -m pip install -r requirements.txt`
- `python main.py`

### Update requirements.txt

- `python -m pip freeze > requirements.txt`