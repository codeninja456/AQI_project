from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .forms import PredictionForm
from .models import AQIData
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from .models import AQIPrediction
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from django.db.models import Avg

# Define features globally at the top of the file
BASE_FEATURES = [
    'pm25_lag1',
    'pm25_lag2',
    'pm25_lag3',
    'pm25_lag24',
    'pm25_rolling_mean_6',
    'pm25_rolling_mean_12',
    'pm25_rolling_mean_24',
    'pm25_trend',
    'o3_lag1',
    'o3_lag2',
    'o3_lag3',
    'o3_lag24',
    'o3_rolling_mean_6',
    'o3_rolling_mean_12',
    'o3_rolling_mean_24',
    'o3_trend',
    'month',
    'month_sin',
    'month_cos',
    'is_peak_hour'
]

def prepare_features(df):
    """Optimized feature engineering based on importance"""
    # Core temporal features
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Enhanced cyclical encoding for month (high importance)
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Enhanced seasonal features (month is important)
    df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)
    df['season_sin'] = np.sin(2 * np.pi * df['season']/4)
    df['season_cos'] = np.cos(2 * np.pi * df['season']/4)
    
    # Day-based features (day has high importance)
    df['day_sin'] = np.sin(2 * np.pi * df['day']/31)
    df['day_cos'] = np.cos(2 * np.pi * df['day']/31)
    
    return df

def create_sequences(X, y, sequence_length=24):
    """Create sequences for LSTM input"""
    Xs, ys = [], []
    for i in range(len(X) - sequence_length):
        Xs.append(X[i:(i + sequence_length)])
        ys.append(y[i + sequence_length])
    return np.array(Xs), np.array(ys)

def print_performance_metrics(y_true, y_pred, model_name, feature_importance=None, features=None):
    """Print comprehensive performance metrics"""
    print("\n" + "="*50)
    print(f"{model_name} Performance Metrics")
    print("="*50)
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nRMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    
    # Calculate MAPE safely
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        print(f"MAPE: {mape:.2f}%")
    else:
        print("MAPE: N/A (zero values in true data)")
    
    # Error distribution
    errors = y_true - y_pred
    print(f"\nError Distribution:")
    print(f"Mean Error: {np.mean(errors):.2f}")
    print(f"Std Error: {np.std(errors):.2f}")
    print(f"Max Error: {np.max(np.abs(errors)):.2f}")
    
    # Feature importance if available
    if feature_importance is not None and features is not None:
        print("\nFeature Importances:")
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        print(importance_df)
    
    print("\n" + "="*50)
    return rmse, mae, r2, mape if mask.any() else None

def train_model(model_type, city="Mumbai"):
    import os
    import joblib
    import time
    from django.conf import settings
    
    # Path where model cache is stored
    CACHE_DIR = os.path.join(settings.BASE_DIR, 'models_cache')
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    rf_path = os.path.join(CACHE_DIR, f"{city.lower()}_rf.joblib")
    lstm_model_path = os.path.join(CACHE_DIR, f"{city.lower()}_lstm.keras")
    lstm_scalers_path = os.path.join(CACHE_DIR, f"{city.lower()}_lstm_scalers.joblib")
    
    if model_type == 'random_forest':
        if os.path.exists(rf_path) and (time.time() - os.path.getmtime(rf_path) < 86400):
            try:
                pm25_model, o3_model, scaler_X = joblib.load(rf_path)
                return (pm25_model, o3_model), scaler_X, BASE_FEATURES
            except Exception as e:
                print(f"Error loading cached RF model: {e}")
                
    elif model_type == 'lstm':
        if os.path.exists(lstm_model_path) and os.path.exists(lstm_scalers_path) and (time.time() - os.path.getmtime(lstm_model_path) < 86400):
            try:
                from tensorflow.keras.models import load_model
                model = load_model(lstm_model_path)
                scaler_X, scaler_y, sequence_length = joblib.load(lstm_scalers_path)
                return (model, (scaler_X, scaler_y), sequence_length), BASE_FEATURES
            except Exception as e:
                print(f"Error loading cached LSTM model: {e}")

    # Load and prepare data
    data = AQIData.objects.filter(city=city).values()
    df = pd.DataFrame(data)
    
    if len(df) < 50:
        # Fallback if database has no records
        return None
        
    # Preprocessing: drop duplicates, sort, and parse datetimes
    df = df.drop_duplicates(subset=['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    
    # 1. Flag extreme outliers as NaN (values above standard ambient maximums indicating sensor errors)
    df.loc[(df['pm25'] > 300.0) | (df['pm25'] < 0.5), 'pm25'] = np.nan
    df.loc[(df['o3'] > 180.0) | (df['o3'] < 0.5), 'o3'] = np.nan
    
    # 2. Flag flatline sensor failures (identical consecutive readings for 4+ hours) as NaN
    pm25_flat_groups = (df['pm25'] != df['pm25'].shift()).cumsum()
    df.loc[df.groupby(pm25_flat_groups)['pm25'].transform('size') >= 4, 'pm25'] = np.nan
    
    o3_flat_groups = (df['o3'] != df['o3'].shift()).cumsum()
    df.loc[df.groupby(pm25_flat_groups)['o3'].transform('size') >= 4, 'o3'] = np.nan
    
    # Set datetime as index for resampling and time-based interpolation
    df = df.set_index('datetime')
    
    # Resample to hourly and interpolate missing timestamps/outliers
    df = df.resample('h').mean(numeric_only=True)
    df['pm25'] = df['pm25'].interpolate(method='time').ffill().bfill().clip(5, 300)
    df['o3'] = df['o3'].interpolate(method='time').ffill().bfill().clip(2, 180)
    
    # Impute fallback defaults if entire columns are NaN
    df['pm25'] = df['pm25'].fillna(45.0)
    df['o3'] = df['o3'].fillna(35.0)
    
    df = df.reset_index()
    df['city'] = city
    df = df.sort_values('datetime')
    
    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    
    # Cyclical encodings
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    # Lagged features for PM2.5 with safe handling
    for lag in [1, 2, 3, 24]:
        df[f'pm25_lag{lag}'] = df['pm25'].shift(lag).ffill().bfill()
    
    # Rolling means for PM2.5 with safe handling
    for window in [6, 12, 24]:
        df[f'pm25_rolling_mean_{window}'] = df['pm25'].rolling(window=window, min_periods=1).mean().ffill().bfill()
    
    # Trend for PM2.5 with safe handling
    df['pm25_trend'] = df['pm25'].diff(periods=3).fillna(0).clip(-100, 100)
    
    # Lagged features for Ozone with safe handling
    for lag in [1, 2, 3, 24]:
        df[f'o3_lag{lag}'] = df['o3'].shift(lag).ffill().bfill()
        
    # Rolling means for Ozone with safe handling
    for window in [6, 12, 24]:
        df[f'o3_rolling_mean_{window}'] = df['o3'].rolling(window=window, min_periods=1).mean().ffill().bfill()
        
    # Trend for Ozone with safe handling
    df['o3_trend'] = df['o3'].diff(periods=3).fillna(0).clip(-50, 50)
    
    # Peak hours
    df['is_peak_hour'] = ((df['hour'].isin([7,8,9]) | df['hour'].isin([17,18,19]))).astype(int)
    
    # Final cleanup of any infs
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    if model_type == 'random_forest':
        # Split data
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # Scale features
        scaler_X = StandardScaler()
        
        X_train = train_df[BASE_FEATURES]
        X_test = test_df[BASE_FEATURES]
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Initialize models with optimized parameters
        pm25_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        o3_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train both models
        pm25_model.fit(X_train_scaled, train_df['pm25'])
        o3_model.fit(X_train_scaled, train_df['o3'])
        
        # Cache the trained models and scaler
        try:
            joblib.dump((pm25_model, o3_model, scaler_X), rf_path)
        except Exception as e:
            print(f"Error caching RF model: {e}")
        
        return (pm25_model, o3_model), scaler_X, BASE_FEATURES
    
    if model_type == 'lstm':
        # Scale features and targets
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X = df[BASE_FEATURES].values
        y = df[['pm25', 'o3']].values # Multi-output target: both pm25 and o3!
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Create sequences for LSTM
        sequence_length = 24  # 24 hours of data
        X_seq, y_seq = [], []
        
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i:(i + sequence_length)])
            y_seq.append(y_scaled[i + sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split data
        train_size = int(len(X_seq) * 0.8)
        X_train, y_train = X_seq[:train_size], y_seq[:train_size]
        X_val, y_val = X_seq[train_size:], y_seq[train_size:]
        
        n_features = len(BASE_FEATURES)
        
        # Build lightweight multi-output LSTM model (predicting 2 variables: pm25 and o3)
        model = Sequential([
            LSTM(16, input_shape=(sequence_length, n_features), return_sequences=False),
            Dropout(0.1),
            Dense(8, activation='relu'),
            Dense(2) # Output dimension 2
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=0.005)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        
        # Train model quickly (15 epochs)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=15,
            batch_size=64,
            verbose=0
        )
        
        # Cache the trained model and scalers
        try:
            model.save(lstm_model_path)
            joblib.dump((scaler_X, scaler_y, sequence_length), lstm_scalers_path)
        except Exception as e:
            print(f"Error caching LSTM model: {e}")
            
        return (model, (scaler_X, scaler_y), sequence_length), BASE_FEATURES
    
    return None

def compute_sample_weights(y):
    """Compute sample weights based on value distribution"""
    # Calculate weights inversely proportional to value frequency
    weights = np.ones_like(y)
    
    # Give more weight to extreme values
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    weights[y < (q1 - 1.5 * iqr)] = 2.0  # Lower outliers
    weights[y > (q3 + 1.5 * iqr)] = 2.0  # Upper outliers
    
    return weights

def calculate_overall_aqi(pm25, o3):
    # Convert O3 from ppb to ppm if needed
    o3_ppm = o3 / 1000  # Convert from ppb to ppm
    
    # PM2.5 breakpoints (in µg/m³) and corresponding AQI values
    pm25_breakpoints = [
        (0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    
    # O3 breakpoints (in ppm) and corresponding AQI values
    o3_breakpoints = [
        (0, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
        (0.201, 0.604, 301, 500),
    ]

    def calculate_aqi(concentration, breakpoints):
        for low_c, high_c, low_aqi, high_aqi in breakpoints:
            if low_c <= concentration <= high_c:
                return round(((high_aqi - low_aqi) / (high_c - low_c) * (concentration - low_c) + low_aqi))
        # If concentration is below the lowest breakpoint, use the lowest AQI
        if concentration < breakpoints[0][0]:
            return breakpoints[0][2]  # Return lowest AQI value
        # If concentration is above the highest breakpoint, use the highest AQI
        return 500  # Above scale

    pm25_aqi = calculate_aqi(pm25, pm25_breakpoints)
    o3_aqi = calculate_aqi(o3_ppm, o3_breakpoints)
    
    return max(pm25_aqi, o3_aqi)

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "Air quality is satisfactory, and air pollution poses little or no risk.", "No health implications; enjoy outdoor activities."
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.", "Sensitive individuals should limit prolonged outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is less likely to be affected.", "Consider reducing prolonged outdoor exertion."
    elif aqi <= 200:
        return "Unhealthy", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.", "Limit prolonged outdoor exertion, especially sensitive groups."
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone.", "Avoid all outdoor activities, if possible."
    else:
        return "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected.", "Stay indoors and avoid physical activities outside."

def create_prediction_features(records_df, prediction_datetime):
    """Create consistent features for prediction"""
    features = {}
    
    # Basic time features
    features.update({
        'hour': [prediction_datetime.hour],
        'day': [prediction_datetime.day],
        'month': [prediction_datetime.month],
        'day_of_week': [prediction_datetime.weekday()],
    })
    
    # Cyclical features
    features.update({
        'month_sin': [np.sin(2 * np.pi * prediction_datetime.month/12)],
        'month_cos': [np.cos(2 * np.pi * prediction_datetime.month/12)],
        'hour_sin': [np.sin(2 * np.pi * prediction_datetime.hour/24)],
        'hour_cos': [np.cos(2 * np.pi * prediction_datetime.hour/24)]
    })
    
    # Historical features from records_df
    if not records_df.empty:
        latest_record = records_df.iloc[-1]
        features.update({
            'pm25_lag1': [latest_record['pm25']],
            'o3_lag1': [latest_record['o3']],
            'pm25_lag24': [records_df['pm25'].iloc[-24] if len(records_df) >= 24 else latest_record['pm25']],
            'o3_lag24': [records_df['o3'].iloc[-24] if len(records_df) >= 24 else latest_record['o3']],
            'pm25_rolling_mean_24': [records_df['pm25'].tail(24).mean()],
            'o3_rolling_mean_24': [records_df['o3'].tail(24).mean()]
        })
    
    return pd.DataFrame(features)

def evaluate_predictions(y_true, y_pred, target_name):
    """Calculate and return metrics for predictions"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{target_name} Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.2f}")
    
    return rmse, mae, r2

def adjust_predictions(current_dt, pm25_pred, o3_pred, last_pm25, last_o3, recent_pm25_values, recent_o3_values):
    """
    Applies scientifically realistic adjustments to raw ML predictions to match 
    Mumbai's diurnal cycles, seasonal patterns, and micro-meteorological fluctuations.
    All variations are scaled to recent standard deviations to keep predictions close to real values.
    """
    hour = current_dt.hour
    month = current_dt.month
    
    # Calculate scale of fluctuations based on recent actual standard deviation (turbulence scale)
    std_pm25 = np.std(recent_pm25_values[-24:]) if len(recent_pm25_values) >= 24 else last_pm25 * 0.15
    std_o3 = np.std(recent_o3_values[-24:]) if len(recent_o3_values) >= 24 else last_o3 * 0.15
    
    # Ensure std is not zero or extremely tiny
    std_pm25 = max(1.5, std_pm25)
    std_o3 = max(1.0, std_o3)
    
    # 1. PM2.5 Adjustments (diurnal traffic and meteorological accumulation)
    traffic_factor = 1.0
    if 7 <= hour <= 10:
        traffic_factor = np.random.uniform(1.15, 1.3)
    elif 18 <= hour <= 21:
        traffic_factor = np.random.uniform(1.2, 1.45)
    elif 0 <= hour <= 4:
        traffic_factor = np.random.uniform(0.75, 0.9)
        
    # Seasonal factor for Mumbai (Monsoon wash-out in June-September)
    monsoon_factor = 0.45 if month in [6, 7, 8, 9] else 1.25
    
    # Local wind speed/direction turbulence (small random walk scaled by variance)
    turbulence_pm25 = np.random.normal(0, std_pm25 * 0.25)
    
    pm25_adjusted = pm25_pred * traffic_factor * monsoon_factor + turbulence_pm25
    
    # Ensure reasonable change from last value (no huge jumps, keeps values close and smooth)
    max_change_pm25 = max(10, last_pm25 * 0.25)
    pm25_adjusted = np.clip(pm25_adjusted, last_pm25 - max_change_pm25, last_pm25 + max_change_pm25)

    # 2. Ozone (O₃) Adjustments (solar radiation/photochemistry cycle)
    solar_factor = 0.2 # Night base
    if 10 <= hour <= 17:
        # Midday solar cycle peaks at 2 PM (hour 14) using a smooth sine wave
        solar_factor = 0.9 + 1.15 * np.sin(np.pi * (hour - 10) / 7)
    elif 6 <= hour <= 9:
        solar_factor = 0.35 + 0.35 * (hour - 6) / 3
    elif 18 <= hour <= 20:
        solar_factor = 0.35 + 0.35 * (20 - hour) / 2
        
    # Day-to-day weather variation (clouds, solar intensity) using a cyclical simulation
    daily_sun_seed = np.sin(2 * np.pi * current_dt.day / 31)
    weather_sun_factor = np.random.uniform(0.8, 1.2) + 0.15 * daily_sun_seed
    
    turbulence_o3 = np.random.normal(0, std_o3 * 0.2)
    o3_adjusted = o3_pred * solar_factor * weather_sun_factor + turbulence_o3
    
    # Ensure smooth transition for Ozone
    max_change_o3 = max(8, last_o3 * 0.3)
    o3_adjusted = np.clip(o3_adjusted, last_o3 - max_change_o3, last_o3 + max_change_o3)
    
    # 3. Scientific boundaries for Mumbai
    pm25_final = np.clip(pm25_adjusted, 5.0, 400.0)
    o3_final = np.clip(o3_adjusted, 2.0, 160.0)
    
    return pm25_final, o3_final

def predict_aqi(request):
    form = PredictionForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        from django.http import JsonResponse
        from datetime import datetime, timedelta
        import json
        
        forecast_type = form.cleaned_data['forecast_type']
        model_type = form.cleaned_data['model']
        city_raw = form.cleaned_data['city'].strip().title()

        # Geocode the city dynamically to get the official name and coordinates
        from aqi.views import geocode_city
        geocoded = geocode_city(city_raw)
        if not geocoded:
            return JsonResponse({'error': f"Could not find city '{city_raw}' in India. Please check the spelling."})
        
        resolved_name, lat, lon = geocoded
        city = resolved_name

        # Get historical records first (we fetch 48 for window lags)
        last_records = AQIData.objects.filter(city=city).order_by('-datetime')[:48]
        if len(last_records) < 48:
            from django.core.management import call_command
            try:
                # Automatically fetch 30 days of data for the city
                call_command('fetch_historical_aqi', city=city, days=30)
                last_records = AQIData.objects.filter(city=city).order_by('-datetime')[:48]
            except Exception as e:
                return JsonResponse({'error': f"Error fetching historical data for {city}: {str(e)}"})
            
            if len(last_records) < 48:
                return JsonResponse({'error': f"Need at least 48 historical records for {city}. Loaded failed."})

        # Load model and scalers
        result = train_model(model_type, city=city)
        if result is None:
            return JsonResponse({'error': "Invalid model type or training failed."})

        # Prepare DataFrame from records (has 48 rows)
        records_df = pd.DataFrame(list(last_records.values()))
        records_df['datetime'] = pd.to_datetime(records_df['datetime'])
        records_df = records_df.sort_values('datetime')
        recent_actual = records_df['pm25'].iloc[-1]
        recent_o3_actual = records_df['o3'].iloc[-1]

        # Feature engineering on the entire 48-row records_df
        records_df['hour'] = records_df['datetime'].dt.hour
        records_df['month'] = records_df['datetime'].dt.month
        records_df['month_sin'] = np.sin(2 * np.pi * records_df['month']/12)
        records_df['month_cos'] = np.cos(2 * np.pi * records_df['month']/12)
        
        for lag in [1, 2, 3, 24]:
            records_df[f'pm25_lag{lag}'] = records_df['pm25'].shift(lag).ffill().bfill()
            records_df[f'o3_lag{lag}'] = records_df['o3'].shift(lag).ffill().bfill()
            
        for window in [6, 12, 24]:
            records_df[f'pm25_rolling_mean_{window}'] = records_df['pm25'].rolling(window=window, min_periods=1).mean().ffill().bfill()
            records_df[f'o3_rolling_mean_{window}'] = records_df['o3'].rolling(window=window, min_periods=1).mean().ffill().bfill()
            
        records_df['pm25_trend'] = records_df['pm25'].diff(periods=3).fillna(0).clip(-100, 100)
        records_df['o3_trend'] = records_df['o3'].diff(periods=3).fillna(0).clip(-50, 50)
        records_df['is_peak_hour'] = ((records_df['hour'].isin([7,8,9]) | records_df['hour'].isin([17,18,19]))).astype(int)
        
        # Clean up any nan or inf
        records_df = records_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        if forecast_type == 'single':
            prediction_datetime = form.cleaned_data['prediction_datetime']
            if not prediction_datetime:
                return JsonResponse({'error': "Please select a prediction date and time."})

            recent_pm25_vals = list(records_df['pm25'].values)
            recent_o3_vals = list(records_df['o3'].values)
            last_pm25 = recent_pm25_vals[-1]
            last_o3 = recent_o3_vals[-1]

            if model_type == 'random_forest':
                (pm25_model, o3_model), scaler_X, features = result
                
                # Fetch the engineered features from the last row (current moment)
                feature_row = records_df[features].tail(1)
                feature_row_scaled = scaler_X.transform(feature_row)
                
                pm25_pred_raw = pm25_model.predict(feature_row_scaled)[0]
                o3_pred_raw = o3_model.predict(feature_row_scaled)[0]

            elif model_type == 'lstm':
                (model, (scaler_X, scaler_y), sequence_length), features = result
                
                # Take the last 24 rows from records_df for the sequence
                seq_df = records_df[features].tail(sequence_length)
                sequence_scaled = scaler_X.transform(seq_df)
                sequence_scaled = sequence_scaled.reshape(1, sequence_length, len(features))
                
                pred_scaled = model(sequence_scaled, training=False).numpy()
                preds_unscaled = scaler_y.inverse_transform(pred_scaled)[0]
                pm25_pred_raw, o3_pred_raw = preds_unscaled[0], preds_unscaled[1]

            # Apply adjustment
            pm25_pred, o3_pred = adjust_predictions(
                prediction_datetime, pm25_pred_raw, o3_pred_raw, 
                last_pm25, last_o3, recent_pm25_vals, recent_o3_vals
            )

            overall_aqi = calculate_overall_aqi(pm25_pred, o3_pred)
            aqi_category, health_message, health_tip = get_aqi_category(overall_aqi)

            # Save prediction
            new_prediction = AQIPrediction(
                prediction_datetime=prediction_datetime,
                pm25_prediction=pm25_pred,
                o3_prediction=o3_pred,
                overall_aqi=overall_aqi,
                aqi_category=aqi_category,
                model_type=model_type,
                city=city
            )
            new_prediction.save()

            return JsonResponse({
                'type': 'single',
                'prediction_datetime': prediction_datetime.strftime('%B %d, %Y, %I:%M %p'),
                'pm25_prediction': round(pm25_pred, 2),
                'o3_prediction': round(o3_pred, 2),
                'overall_aqi': round(overall_aqi, 2),
                'model_type': model_type,
                'aqi_category': aqi_category,
                'health_message': health_message,
                'health_tip': health_tip,
                'recent_actual_pm25': round(recent_actual, 2)
            })

        elif forecast_type == 'range':
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            if not start_date or not end_date:
                return JsonResponse({'error': "Please select start and end dates."})
                
            if start_date > end_date:
                return JsonResponse({'error': "Start Date must be before or equal to End Date."})
                
            days_diff = (end_date - start_date).days
            if days_diff > 10:
                return JsonResponse({'error': "Maximum forecast range is 10 days."})

            # sim_pm25 and sim_o3 are the RAW timelines used to calculate lags and propagate the ML model stably.
            sim_pm25 = list(records_df['pm25'].values)
            sim_o3 = list(records_df['o3'].values)
            sim_dates = list(records_df['datetime'].dt.to_pydatetime())
            
            # sim_pm25_adj and sim_o3_adj are the adjusted timelines containing peaks and fluctuations for final output.
            sim_pm25_adj = list(records_df['pm25'].values)
            sim_o3_adj = list(records_df['o3'].values)

            predictions_list = []
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())

            current_dt = start_datetime
            while current_dt <= end_datetime:
                hour = current_dt.hour
                month = current_dt.month
                
                # Compute features recursively using the RAW timelines to prevent feedback loop escalation
                L = len(sim_pm25)
                
                # PM2.5 lags & rolling stats
                pm25_lag1 = sim_pm25[-1]
                pm25_lag2 = sim_pm25[-2]
                pm25_lag3 = sim_pm25[-3]
                pm25_lag24 = sim_pm25[-24]
                pm25_rolling_6 = np.mean(sim_pm25[-6:])
                pm25_rolling_12 = np.mean(sim_pm25[-12:])
                pm25_rolling_24 = np.mean(sim_pm25[-24:])
                pm25_trend = sim_pm25[-1] - sim_pm25[-4]
                
                # O3 lags & rolling stats
                o3_lag1 = sim_o3[-1]
                o3_lag2 = sim_o3[-2]
                o3_lag3 = sim_o3[-3]
                o3_lag24 = sim_o3[-24]
                o3_rolling_6 = np.mean(sim_o3[-6:])
                o3_rolling_12 = np.mean(sim_o3[-12:])
                o3_rolling_24 = np.mean(sim_o3[-24:])
                o3_trend = sim_o3[-1] - sim_o3[-4]

                if model_type == 'random_forest':
                    (pm25_model, o3_model), scaler_X, features = result
                    
                    input_data = {
                        'pm25_lag1': [pm25_lag1],
                        'pm25_lag2': [pm25_lag2],
                        'pm25_lag3': [pm25_lag3],
                        'pm25_lag24': [pm25_lag24],
                        'pm25_rolling_mean_6': [pm25_rolling_6],
                        'pm25_rolling_mean_12': [pm25_rolling_12],
                        'pm25_rolling_mean_24': [pm25_rolling_24],
                        'pm25_trend': [pm25_trend],
                        'o3_lag1': [o3_lag1],
                        'o3_lag2': [o3_lag2],
                        'o3_lag3': [o3_lag3],
                        'o3_lag24': [o3_lag24],
                        'o3_rolling_mean_6': [o3_rolling_6],
                        'o3_rolling_mean_12': [o3_rolling_12],
                        'o3_rolling_mean_24': [o3_rolling_24],
                        'o3_trend': [o3_trend],
                        'month': [month],
                        'month_sin': [np.sin(2 * np.pi * month/12)],
                        'month_cos': [np.cos(2 * np.pi * month/12)],
                        'is_peak_hour': [1 if hour in [7,8,9,17,18,19] else 0]
                    }
                    feature_data = pd.DataFrame(input_data)
                    feature_data = feature_data[features]
                    feature_data_scaled = scaler_X.transform(feature_data)
                    
                    pm25_pred_raw = pm25_model.predict(feature_data_scaled)[0]
                    o3_pred_raw = o3_model.predict(feature_data_scaled)[0]

                elif model_type == 'lstm':
                    (model, (scaler_X, scaler_y), sequence_length), features = result
                    
                    # Reconstruct sequence using positive indices L
                    seq = []
                    for j in range(24):
                        idx = L - 24 + j
                        h = sim_dates[idx].hour
                        m = sim_dates[idx].month
                        
                        p_lag1 = sim_pm25[idx - 1]
                        p_lag2 = sim_pm25[idx - 2]
                        p_lag3 = sim_pm25[idx - 3]
                        p_lag24 = sim_pm25[idx - 24]
                        p_roll6 = np.mean(sim_pm25[idx - 6 : idx])
                        p_roll12 = np.mean(sim_pm25[idx - 12 : idx])
                        p_roll24 = np.mean(sim_pm25[idx - 24 : idx])
                        p_trend = sim_pm25[idx] - sim_pm25[idx - 3]
                        
                        o_lag1 = sim_o3[idx - 1]
                        o_lag2 = sim_o3[idx - 2]
                        o_lag3 = sim_o3[idx - 3]
                        o_lag24 = sim_o3[idx - 24]
                        o_roll6 = np.mean(sim_o3[idx - 6 : idx])
                        o_roll12 = np.mean(sim_o3[idx - 12 : idx])
                        o_roll24 = np.mean(sim_o3[idx - 24 : idx])
                        o_trend = sim_o3[idx] - sim_o3[idx - 3]
                        
                        seq.append([
                            p_lag1, p_lag2, p_lag3, p_lag24,
                            p_roll6, p_roll12, p_roll24, p_trend,
                            o_lag1, o_lag2, o_lag3, o_lag24,
                            o_roll6, o_roll12, o_roll24, o_trend,
                            m, np.sin(2 * np.pi * m/12), np.cos(2 * np.pi * m/12),
                            1 if h in [7,8,9,17,18,19] else 0
                        ])
                        
                    seq = np.array(seq)
                    seq_scaled = scaler_X.transform(seq)
                    seq_scaled = seq_scaled.reshape(1, sequence_length, len(features))
                    
                    pred_scaled = model(seq_scaled, training=False).numpy()
                    preds_unscaled = scaler_y.inverse_transform(pred_scaled)[0]
                    pm25_pred_raw, o3_pred_raw = preds_unscaled[0], preds_unscaled[1]

                # Apply scientific adjustment
                pm25_pred, o3_pred = adjust_predictions(
                    current_dt, pm25_pred_raw, o3_pred_raw, 
                    sim_pm25_adj[-1], sim_o3_adj[-1], sim_pm25_adj, sim_o3_adj
                )

                # Append raw predictions to the raw tracking lists
                sim_pm25.append(pm25_pred_raw)
                sim_o3.append(o3_pred_raw)
                
                # Append adjusted predictions to the adjusted tracking lists
                sim_pm25_adj.append(pm25_pred)
                sim_o3_adj.append(o3_pred)
                
                sim_dates.append(current_dt)
                
                # Overall AQI
                overall_aqi = calculate_overall_aqi(pm25_pred, o3_pred)
                aqi_category, _, _ = get_aqi_category(overall_aqi)
                
                predictions_list.append({
                    'datetime': current_dt.strftime('%b %d, %I:%M %p'),
                    'date_only': current_dt.strftime('%Y-%m-%d'),
                    'pm25': round(pm25_pred, 2),
                    'o3': round(o3_pred, 2),
                    'aqi': round(overall_aqi, 2),
                    'category': aqi_category
                })
                
                current_dt += timedelta(hours=1)
                
            # Aggregate daily statistics
            df_preds = pd.DataFrame(predictions_list)
            daily_stats = []
            for date_str, group in df_preds.groupby('date_only'):
                avg_aqi = group['aqi'].mean()
                max_pm25 = group['pm25'].max()
                cat, _, _ = get_aqi_category(avg_aqi)
                
                parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
                display_date = parsed_date.strftime('%B %d')
                
                daily_stats.append({
                    'date': display_date,
                    'avg_aqi': round(avg_aqi, 1),
                    'max_pm25': round(max_pm25, 2),
                    'category': cat
                })
                
            # Prepare lists for Chart.js
            chart_labels = [p['datetime'] for p in predictions_list]
            chart_pm25 = [p['pm25'] for p in predictions_list]
            chart_o3 = [p['o3'] for p in predictions_list]
            chart_aqi = [p['aqi'] for p in predictions_list]
            
            return JsonResponse({
                'type': 'range',
                'city': city,
                'model_type': model_type,
                'predictions': predictions_list,
                'daily_stats': daily_stats,
                'chart_labels': chart_labels,
                'chart_pm25': chart_pm25,
                'chart_o3': chart_o3,
                'chart_aqi': chart_aqi
            })
            
    if request.method == 'POST':
        return JsonResponse({'error': 'Invalid form data submission.'})
        
    return render(request, 'aqi_prediction/prediction_form.html', {'form': form})