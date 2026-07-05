from django.shortcuts import render
import requests
from django.conf import settings

from django.shortcuts import render
from django.contrib.auth.decorators import login_required

# from django.http import HttpResponse

# def home(request):
#     return render(request,'website\\index.html')
#     # return HttpResponse('works')          for testing

def about(request):
    return render(request,'website//about.html')

# weather data fetch
def geocode_city(city_name):
    api_key = settings.WEATHER_API_KEY
    # Restrict to India (IN)
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name},IN&limit=1&appid={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]['name'], data[0]['lat'], data[0]['lon']
    except Exception:
        pass
    return None

def get_weather_data(lat, lon):
    api_key = settings.WEATHER_API_KEY
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# aqi data fetch
def get_aqi_data(lat, lon):
    api_key = settings.WEATHER_API_KEY
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
@login_required
def home(request):
    selected_city = request.GET.get('city', 'Mumbai').strip()
    if not selected_city:
        selected_city = "Mumbai"
        
    geocoded = geocode_city(selected_city)
    if geocoded:
        resolved_name, lat, lon = geocoded
    else:
        # Fallback to Mumbai
        resolved_name, lat, lon = "Mumbai", 19.0760, 72.8777
        
    weather_data = get_weather_data(lat, lon)
    aqi_data = get_aqi_data(lat, lon)
    
    context = {
        'weather': weather_data,
        'aqi': aqi_data,
        'selected_city': resolved_name,
    }
    return render(request, 'website//index.html', context)



