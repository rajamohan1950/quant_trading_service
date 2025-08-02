INTERVAL_OPTIONS = {
    '5minute': '5minute',
    '15minute': '15minute',
    '30minute': '30minute',
    '60minute': '60minute',
    'day': 'day'
}

def is_valid_interval(interval):
    return interval in INTERVAL_OPTIONS

def get_interval_label(interval):
    return INTERVAL_OPTIONS.get(interval, interval) 