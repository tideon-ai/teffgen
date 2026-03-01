"""
Weather API tool with free Open-Meteo backend.

Primary backend: Open-Meteo API (100% free, no API key required)
Optional backend: OpenWeatherMap (free tier, needs API key)
"""

import json
import logging
import time
from typing import Any, Dict, Optional
from urllib.request import urlopen, Request
from urllib.parse import urlencode, quote_plus
from urllib.error import URLError


def _get_user_agent() -> str:
    try:
        from effgen import __version__
    except ImportError:
        __version__ = "dev"
    return f"effGen/{__version__}"

from ..base_tool import (
    BaseTool,
    ToolCategory,
    ToolMetadata,
    ParameterSpec,
    ParameterType,
)

logger = logging.getLogger(__name__)


class WeatherTool(BaseTool):
    """
    Weather information tool using free APIs.

    Primary backend: Open-Meteo (free, no key required)
    Optional backend: OpenWeatherMap (free tier, needs key)

    Features:
    - Current weather conditions
    - Temperature, humidity, wind speed
    - Weather description
    - Geocoding (city name to coordinates)
    - Metric and imperial units
    - 1-hour result caching
    """

    # WMO Weather Code descriptions
    WMO_CODES = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
        77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
        82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
        95: "Thunderstorm", 96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }

    def __init__(
        self,
        backend: str = "open_meteo",
        api_key: Optional[str] = None,
        cache_ttl: int = 3600,
    ):
        """
        Initialize the Weather tool.

        Args:
            backend: Backend to use ("open_meteo" or "openweathermap")
            api_key: API key for OpenWeatherMap (not needed for Open-Meteo)
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        super().__init__(
            metadata=ToolMetadata(
                name="weather",
                description=(
                    "Get current weather information for a location. "
                    "Returns temperature, humidity, wind speed, and conditions. "
                    "Uses the free Open-Meteo API by default (no API key needed)."
                ),
                category=ToolCategory.EXTERNAL_API,
                parameters=[
                    ParameterSpec(
                        name="location",
                        type=ParameterType.STRING,
                        description="City name or location (e.g., 'New York', 'London, UK', 'Tokyo')",
                        required=True,
                        min_length=1,
                        max_length=200,
                    ),
                    ParameterSpec(
                        name="units",
                        type=ParameterType.STRING,
                        description="Temperature units: 'metric' (Celsius) or 'imperial' (Fahrenheit)",
                        required=False,
                        default="metric",
                        enum=["metric", "imperial"],
                    ),
                ],
                returns={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "temperature": {"type": "number"},
                        "units": {"type": "string"},
                        "humidity": {"type": "number"},
                        "wind_speed": {"type": "number"},
                        "conditions": {"type": "string"},
                    },
                },
                timeout_seconds=15,
                tags=["weather", "api", "free", "open-meteo"],
                examples=[
                    {
                        "location": "New York",
                        "output": {
                            "location": "New York",
                            "temperature": 22.5,
                            "units": "celsius",
                            "humidity": 65,
                            "wind_speed": 12.3,
                            "conditions": "Partly cloudy",
                        },
                    },
                ],
            )
        )

        self.backend = backend
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}  # key -> (timestamp, data)

        if backend == "openweathermap" and api_key:
            logger.warning(
                "\u26a0\ufe0f  OpenWeatherMap backend requires a free API key from openweathermap.org.\n"
                "    Using Open-Meteo (free, no key required) as default."
            )

    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached result if not expired."""
        if key in self._cache:
            ts, data = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Dict):
        """Cache a result."""
        self._cache[key] = (time.time(), data)

    def _fetch_url(self, url: str) -> Dict:
        """Fetch JSON from a URL."""
        req = Request(url, headers={"User-Agent": _get_user_agent()})
        try:
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except URLError as e:
            raise ConnectionError(f"Failed to fetch weather data: {e}")

    def _geocode(self, location: str) -> tuple:
        """
        Convert location name to coordinates using Open-Meteo geocoding API.

        Returns:
            (latitude, longitude, display_name)
        """
        cache_key = f"geo:{location.lower()}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached["lat"], cached["lon"], cached["name"]

        params = urlencode({"name": location, "count": 1, "language": "en", "format": "json"})
        url = f"https://geocoding-api.open-meteo.com/v1/search?{params}"

        data = self._fetch_url(url)
        results = data.get("results", [])
        if not results:
            raise ValueError(f"Location not found: '{location}'")

        r = results[0]
        lat, lon = r["latitude"], r["longitude"]
        name = r.get("name", location)
        country = r.get("country", "")
        display_name = f"{name}, {country}" if country else name

        self._set_cache(cache_key, {"lat": lat, "lon": lon, "name": display_name})
        return lat, lon, display_name

    def _fetch_open_meteo(self, lat: float, lon: float, units: str) -> Dict:
        """Fetch weather from Open-Meteo API."""
        temp_unit = "fahrenheit" if units == "imperial" else "celsius"
        wind_unit = "mph" if units == "imperial" else "kmh"

        params = urlencode({
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "temperature_unit": temp_unit,
            "wind_speed_unit": wind_unit,
        })
        url = f"https://api.open-meteo.com/v1/forecast?{params}"

        return self._fetch_url(url)

    async def _execute(
        self,
        location: str,
        units: str = "metric",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Get weather for a location.

        Args:
            location: City name or location string.
            units: 'metric' or 'imperial'.

        Returns:
            Dict with weather information.
        """
        cache_key = f"weather:{location.lower()}:{units}"
        cached = self._get_cached(cache_key)
        if cached:
            cached["cached"] = True
            return cached

        # Geocode location
        lat, lon, display_name = self._geocode(location)

        # Fetch weather
        data = self._fetch_open_meteo(lat, lon, units)
        current = data.get("current", {})

        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        weather_code = current.get("weather_code", 0)
        wind_speed = current.get("wind_speed_10m")

        conditions = self.WMO_CODES.get(weather_code, "Unknown")
        temp_unit = "fahrenheit" if units == "imperial" else "celsius"
        wind_unit = "mph" if units == "imperial" else "km/h"
        temp_symbol = "\u00b0F" if units == "imperial" else "\u00b0C"

        result = {
            "location": display_name,
            "temperature": temp,
            "temperature_display": f"{temp}{temp_symbol}",
            "units": temp_unit,
            "humidity": humidity,
            "humidity_display": f"{humidity}%",
            "wind_speed": wind_speed,
            "wind_display": f"{wind_speed} {wind_unit}",
            "conditions": conditions,
            "weather_code": weather_code,
            "coordinates": {"latitude": lat, "longitude": lon},
            "cached": False,
        }

        self._set_cache(cache_key, result)
        return result
