"""
Context processors to make variables available in all templates
"""
from django.conf import settings

def google_maps_key(request):
    """Make map API keys available in templates"""
    return {
        'MAPBOX_API_KEY': getattr(settings, 'MAPBOX_API_KEY', ''),
    }
