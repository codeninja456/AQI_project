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
def get_weather_data(city="Mumbai"):
    api_key = settings.WEATHER_API_KEY
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# aqi data fetch
def get_aqi_data(city="Mumbai"):
    # Latitude and longitude for Mumbai and Thane
    COORDINATES = {
        "Mumbai": (19.0760, 72.8777),
        "Thane": (19.2183, 72.9781)
    }
    lat, lon = COORDINATES.get(city, (19.0760, 72.8777))
    api_key = settings.WEATHER_API_KEY
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"

    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
@login_required
def home(request):
    selected_city = request.GET.get('city', 'Mumbai')
    if selected_city not in ["Mumbai", "Thane"]:
        selected_city = "Mumbai"
        
    weather_data = get_weather_data(selected_city)
    aqi_data = get_aqi_data(selected_city)
    
    context = {
        'weather': weather_data,
        'aqi': aqi_data,
        'selected_city': selected_city,
    }
    return render(request, 'website//index.html', context)



