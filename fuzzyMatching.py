from rapidfuzz import fuzz, process

# Dataset anime (contoh)
anime_titles = [
    "Naruto",
    "Attack on Titan",
    "Death Note",
    "One Piece",
    "My Hero Academia"
]

# Input dari pengguna
user_input = "attack on titan"  # Salah ketik

# Mencari judul anime yang paling mirip
result = process.extractOne(user_input, anime_titles, scorer=fuzz.ratio)

# Ambang batas (threshold) untuk kecocokan
threshold = 70

if result and result[1] >= threshold:
    print(f"Rekomendasi anime berdasarkan pencarian Anda: {result[0]}")
else:
    print("Maaf, anime yang Anda cari tidak ditemukan.")
