#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate
python manage.py fetch_historical_aqi --city Mumbai --days 30
python manage.py fetch_historical_aqi --city Thane --days 30
