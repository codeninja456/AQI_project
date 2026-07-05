import requests
import pytz
from datetime import datetime, timedelta, timezone
from django.core.management.base import BaseCommand
from django.conf import settings
from aqi_prediction.models import AQIData

class Command(BaseCommand):
    help = 'Fetch historical air pollution data from OpenWeatherMap API for Mumbai or Thane'

    def add_arguments(self, parser):
        parser.add_argument('--city', type=str, default='Mumbai', help='City name to fetch data for')
        parser.add_argument('--days', type=int, default=180, help='Number of days of history to fetch (default: 180)')

    def handle(self, *args, **options):
        city_raw = options['city']
        days = options['days']
        
        # Geocode the city dynamically to get the official name and coordinates
        from aqi.views import geocode_city
        geocoded = geocode_city(city_raw)
        if not geocoded:
            self.stdout.write(self.style.ERROR(f"Error: Could not resolve coordinates for city '{city_raw}' in India."))
            return
            
        resolved_name, lat, lon = geocoded
        city = resolved_name
        api_key = settings.WEATHER_API_KEY
        
        if not api_key:
            self.stdout.write(self.style.ERROR("Error: WEATHER_API_KEY is not configured in settings."))
            return
            
        self.stdout.write(self.style.SUCCESS(f"Fetching {days} days of historical air pollution data for {city}..."))
        
        # Define time range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        # OpenWeatherMap history API works best in smaller chunks (e.g. 30 days) to prevent timeouts
        chunk_days = 30
        current_start = start_date
        
        records_to_create = []
        total_fetched = 0
        
        # Delete existing data for the selected city to prevent duplicates
        AQIData.objects.filter(city=city).delete()
        self.stdout.write(self.style.WARNING(f"Cleared existing AQI data for {city}"))
        
        local_tz = pytz.timezone('Asia/Kolkata')
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=chunk_days), end_date)
            
            start_ts = int(current_start.timestamp())
            end_ts = int(current_end.timestamp())
            
            url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={api_key}"
            
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    self.stdout.write(self.style.ERROR(f"Failed to fetch data for chunk {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}. Code: {response.status_code}"))
                    self.stdout.write(response.text)
                    current_start = current_end
                    continue
                
                data = response.json()
                records = data.get('list', [])
                self.stdout.write(f"Fetched {len(records)} hourly records for chunk {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
                
                for item in records:
                    dt_utc = datetime.fromtimestamp(item['dt'], tz=timezone.utc)
                    dt_local = dt_utc.astimezone(local_tz)
                    
                    pm25 = float(item['components'].get('pm2_5', 0))
                    o3 = float(item['components'].get('o3', 0))
                    
                    records_to_create.append(
                        AQIData(
                            datetime=dt_local,
                            pm25=pm25,
                            o3=o3,
                            city=city
                        )
                    )
                
                total_fetched += len(records)
                
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error fetching chunk: {str(e)}"))
                
            current_start = current_end
            
        # Bulk create records in database for efficiency
        if records_to_create:
            self.stdout.write("Saving records to database...")
            AQIData.objects.bulk_create(records_to_create)
            self.stdout.write(self.style.SUCCESS(f"Successfully loaded {len(records_to_create)} records for {city} into the database."))
        else:
            self.stdout.write(self.style.WARNING("No records fetched."))
