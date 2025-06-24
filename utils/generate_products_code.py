import datetime
import random

def generate_product_code(category_prefix="PROD"): # category_prefix jangan menggunakan koma
    """
    Menghasilkan kode produk yang mudah diingat.
    Contoh: PROD-20250619-0001
    """
    # Bagian tanggal
    today = datetime.date.today()
    date_str = today.strftime("%Y%m%d") # Format YYYYMMDD
    sequential_part = f"{random.randint(1, 9999):04d}" # 4 digit angka acak

    # Gabungkan
    return f"{category_prefix}-{date_str}-{sequential_part}"