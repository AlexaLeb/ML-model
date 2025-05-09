import os
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # подхватит .env из корня

API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = "Moscow"
OUTPUT_CSV = "/app/data/weather.csv"


def fetch_weather() -> dict:
    """Вызывает OpenWeatherMap API и возвращает JSON."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": CITY,
        "appid": API_KEY,
        "units": "metric"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def save_weather(raw: dict):
    """Из JSON вытаскивает нужные поля и дозаписывает строку в CSV."""
    # Парсим значения
    dt = datetime.fromtimestamp(raw["dt"]).isoformat()
    city = raw.get("name", CITY)
    weather_main = raw["weather"][0]["main"]
    weather_desc = raw["weather"][0]["description"]
    temp = raw["main"]["temp"]
    feels_like = raw["main"]["feels_like"]
    pressure = raw["main"]["pressure"]
    wind_speed = raw["wind"]["speed"]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    file_exists = os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # если файл новый — пишем заголовок
        if not file_exists:
            writer.writerow([
                "datetime","city","weather_main","weather_description",
                "temp","feels_like","pressure","wind_speed"
            ])
        writer.writerow([
            dt, city, weather_main, weather_desc,
            temp, feels_like, pressure, wind_speed
        ])
    print(f"Appended weather for {dt}")
