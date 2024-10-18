def format_price(price):
    if price < 1:
        return f'{price:.6f}'
    elif price < 10:
        return f'{price:.4f}'
    else:
        return f'{price:.2f}'
