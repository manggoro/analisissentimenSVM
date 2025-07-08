from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd

print("Pastikan semua jendela Chrome sudah DITUTUP sebelum menjalankan script ini!")
print("Jika error, coba ganti profile Chrome di baris options.add_argument('--profile-directory=Default') menjadi 'Profile 1', 'Profile 2', dst.")

# Input link Google Maps dari user
maps_url = input("Masukkan link Google Maps tempat yang ingin di-scrape: ").strip()

options = Options()
# options.add_argument("--headless")  # Nonaktifkan headless agar browser terlihat
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument(r'--user-data-dir=C:/Users/mufid/AppData/Local/Google/Chrome/User Data')
options.add_argument('--profile-directory=Default')  # Ganti ke 'Profile 1' jika error
print("Inisialisasi Chrome driver...")
driver = webdriver.Chrome(options=options)

print("Membuka link:", maps_url)
driver.get(maps_url)
time.sleep(5)

# Klik tombol "Lihat semua ulasan" (jika ada)
try:
    all_reviews_button = driver.find_element(By.XPATH, '//button[contains(@aria-label, "ulasan")]')
    all_reviews_button.click()
    time.sleep(3)
except Exception as e:
    print("Tombol ulasan tidak ditemukan atau sudah terbuka:", e)

# Scroll otomatis dan klik tombol 'Selanjutnya' jika ada
try:
    scrollable_div = driver.find_element(By.XPATH, '//div[@role="region"]')
    last_count = 0
    max_scroll = 1000
    for i in range(max_scroll):
        driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', scrollable_div)
        time.sleep(1.5)
        # Coba klik tombol 'Selanjutnya' atau 'More reviews' jika ada
        try:
            next_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Selanjutnya') or contains(text(), 'More reviews') or contains(text(), 'Berikutnya')]")
            if next_button.is_displayed() and next_button.is_enabled():
                next_button.click()
                print('Klik tombol Selanjutnya/More reviews')
                time.sleep(2)
        except:
            pass
        reviews = driver.find_elements(By.XPATH, '//div[@data-review-id]')
        if len(reviews) == last_count:
            print(f"Stuck at {len(reviews)} reviews, stopping scroll.")
            break
        last_count = len(reviews)
        if (i+1) % 50 == 0:
            print(f"Scrolled {i+1} times, {len(reviews)} reviews loaded...")
except Exception as e:
    print("Gagal scroll:", e)

# Ambil data review dengan selector yang lebih fleksibel
reviews = driver.find_elements(By.XPATH, '//div[@data-review-id]')
data = []
for review in reviews:
    try:
        # Nama reviewer
        try:
            nama = review.find_element(By.XPATH, ".//span[contains(@class, 'd4r55') or contains(@class, 'WNxzHc')]").text
        except:
            nama = ''
        # Rating
        try:
            rating = review.find_element(By.XPATH, ".//span[contains(@class, 'kvMYJc') or contains(@aria-label, 'bintang')]").get_attribute('aria-label')
        except:
            rating = ''
        # Tanggal
        try:
            tanggal = review.find_element(By.XPATH, ".//span[contains(@class, 'rsqaWe') or contains(@class, 'dehysf')]").text
        except:
            tanggal = ''
        # Isi review
        try:
            isi = review.find_element(By.XPATH, ".//span[contains(@class, 'wiI7pd') or contains(@class, 'review-full-text')]").text
        except:
            isi = ''
        data.append({
            'nama': nama,
            'rating': rating,
            'tanggal': tanggal,
            'review': isi
        })
    except Exception as e:
        continue

driver.quit()

df = pd.DataFrame(data)
df.to_csv('google_maps_reviews.csv', index=False, encoding='utf-8-sig')
print(f"Selesai! {len(data)} review tersimpan di google_maps_reviews.csv") 