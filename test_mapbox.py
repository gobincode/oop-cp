import urllib.request
import json
import sys

import os
token = os.environ.get('MAPBOX_API_KEY', '')

print("=== Mapbox API Tests ===")

# Test 1: Geocoding
print("\n[1] Geocoding: Mumbai")
url = 'https://api.mapbox.com/geocoding/v5/mapbox.places/Mumbai.json?access_token=' + token + '&limit=1'
with urllib.request.urlopen(url, timeout=10) as r:
    data = json.loads(r.read())
    if data.get('features'):
        f = data['features'][0]
        print("  PASS - place:", f['place_name'].encode('ascii','replace').decode())
        print("  center:", f['center'])
    else:
        print("  FAIL - no features")

print("\n=== Overpass API (OpenStreetMap) Nearby Search ===")

# Test 2: Overpass API for nearby hospitals
print("\n[2] Nearby hospitals in Mumbai (Overpass)")
query = '[out:json][timeout:10];node["amenity"~"hospital|clinic|doctors"]["name"](around:5000,19.077793,72.87872);out body 6;'
overpass_url = 'https://overpass-api.de/api/interpreter'
req_data = ('data=' + urllib.request.quote(query)).encode()
req = urllib.request.Request(overpass_url, data=req_data, method='POST',
                              headers={'User-Agent': 'HeartDiseasePrediction/1.0', 'Content-Type': 'application/x-www-form-urlencoded'})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.loads(r.read())
        elements = data.get('elements', [])
        print("  Results:", len(elements))
        for e in elements[:5]:
            tags = e.get('tags', {})
            name = tags.get('name', 'Unknown').encode('ascii', 'replace').decode()
            amenity = tags.get('amenity', '')
            lat = e.get('lat', 0)
            lng = e.get('lon', 0)
            print(f"    - [{amenity}] {name} @ ({lat:.4f}, {lng:.4f})")
except Exception as ex:
    print("  FAIL:", ex)
