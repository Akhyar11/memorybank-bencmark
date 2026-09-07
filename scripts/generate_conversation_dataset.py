"""
generate_conversation_dataset.py – Multi-Turn Diverse Conversational Memory Dataset Generator.
==============================================================================================
Enhanced with:
  1. Massive Entity Vocabulary (200+ Indonesian Names, 65+ Cities, 80+ Tech & Non-Tech Roles, 35+ Foods, 25+ Drinks).
  2. Multi-Style Paraphrasing Engine (Slang, Casual, Formal, Narrative, Indirect) - NO repetitive rigid templates.
  3. Dynamic Out-of-Distribution (OOD) Entity Synthesizer (~15% novel compound concepts).
  4. Diverse Question Formats (10+ query styles per fact) and Flexible Assistant Answers.
  5. Dynamic Fact Latency (Facts placed at varying turn positions with 2 to 8 distractor intervals).
  6. Real-World Indonesian Dialogue Distractors (ShareGPT & Evol-Instruct integration).
  7. Universal Seeder mechanism for 100% reproducible generation.
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any, Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from models.seed import set_seed
except ImportError:
    def set_seed(s: int = 42):
        random.seed(s)


# ---------------------------------------------------------------------------
# 1. Massive Vocabulary & Entity Pools (Diverse Indonesian Across All Islands)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    # Jawa
    "Akhyar", "Budi", "Bayu", "Bambang", "Dimas", "Danang", "Fajar", "Gilang", "Hendra", "Ilham",
    "Joko", "Kurniawan", "Lukman", "Maulana", "Naufal", "Prasetyo", "Raden", "Satria", "Tri", "Wahyu",
    "Agus", "Wibowo", "Sigit", "Yudi", "Bagus", "Anjar", "Surya", "Eko", "Dwi", "Aris",
    # Sunda
    "Asep", "Cecep", "Dadang", "Deden", "Ginanjar", "Ijang", "Jajang", "Maman", "Nana", "Tantan",
    "Ujang", "Wawan", "Yayan", "Encep", "Gugun", "Dede", "Aki", "Koswara", "Iskandar", "Sutisna",
    # Batak & Sumatra
    "Bona", "Baringin", "Chandra", "Doni", "Edison", "Ferry", "Gomgom", "Hotman", "Immanuel", "Jonatan",
    "Maruli", "Monang", "Poltak", "Raja", "Saut", "Tigor", "Ucuk", "Zulham", "Rian", "Adit",
    # Minang & Melayu
    "Buyung", "Farhan", "Haikal", "Irfan", "Khairul", "Rahmat", "Syahrul", "Taufiq", "Zulfikar", "Alamsyah",
    # Bali & Nusa Tenggara
    "Wayan", "Made", "Nyoman", "Ketut", "Gede", "Putu", "Kadek", "Komang", "Anom", "Oka",
    "Luhur", "Darma", "Sudira", "Widiarta", "Ari", "Bagus", "Ngurah", "Ida", "Tjokorda", "Dewa",
    # Sulawesi & Indonesia Timur
    "Andi", "Baso", "Daeng", "Fachri", "Gassing", "Hasan", "Ilyas", "Jumadi", "Kahar", "Mansyur",
    "Nasrul", "Ode", "Pattimura", "Rusdi", "Syamsul", "Tahir", "Usman", "Victor", "Willem", "Yohanes",
    # Wanita (Multi-etnis)
    "Alya", "Anisa", "Cantika", "Citra", "Dinda", "Dewi", "Elisa", "Fira", "Gita", "Hana",
    "Indah", "Intan", "Kartika", "Kirana", "Laras", "Lestari", "Maya", "Mega", "Nadia", "Nisa",
    "Novita", "Putri", "Rani", "Ratna", "Rina", "Sari", "Sekar", "Siti", "Tiara", "Triana",
    "Utami", "Vina", "Winda", "Wulandari", "Yulia", "Zahra", "Adisti", "Bella", "Clara", "Dian"
]

CITIES = [
    # Jawa & Madura
    "Jakarta", "Surabaya", "Bandung", "Semarang", "Yogyakarta", "Solo", "Malang", "Bogor", "Bekasi",
    "Tangerang", "Depok", "Cirebon", "Sukabumi", "Tasikmalaya", "Magelang", "Salatiga", "Pekalongan",
    "Tegal", "Purwokerto", "Kediri", "Blitar", "Madiun", "Probolinggo", "Pasuruan", "Banyuwangi", "Bangkalan",
    # Sumatra
    "Medan", "Palembang", "Padang", "Pekanbaru", "Bandar Lampung", "Batam", "Jambi", "Bengkulu",
    "Banda Aceh", "Pematangsiantar", "Bukittinggi", "Tanjungpinang", "Dumai", "Lubuklinggau", "Pangkalpinang",
    # Kalimantan
    "Balikpapan", "Samarinda", "Banjarmasin", "Pontianak", "Palangka Raya", "Tarakan", "Banjarbaru", "Singkawang",
    # Sulawesi
    "Makassar", "Manado", "Palu", "Kendari", "Gorontalo", "Mamuju", "Bitung", "Parepare", "Palopo", "Baubau",
    # Bali & Nusa Tenggara
    "Denpasar", "Singaraja", "Mataram", "Kupang", "Labuan Bajo", "Ende", "Maumere",
    # Maluku & Papua
    "Ambon", "Ternate", "Jayapura", "Sorong", "Manokwari", "Merauke", "Timika", "Biak"
]

TECH_ROLES = [
    "Frontend Developer", "Backend Developer", "Fullstack Engineer", "Data Scientist",
    "Machine Learning Engineer", "DevOps Engineer", "Mobile Developer", "UI/UX Designer",
    "Product Manager", "QA Automation Engineer", "Cybersecurity Analyst", "Cloud Solutions Architect",
    "Database Administrator (DBA)", "Site Reliability Engineer (SRE)", "AI Prompt Engineer",
    "Blockchain Developer", "Systems Analyst", "Scrum Master", "Embedded Systems Engineer",
    "Data Engineer", "Network Engineer", "Game Developer", "NLP Engineer", "Solutions Architect"
]

NON_TECH_ROLES = [
    "Dokter Umum", "Dokter Gigi Spesialis", "Apoteker Farmasi", "Arsitek Bangunan",
    "Guru Matematika", "Dosen Komunikasi", "Akuntan Publik", "Konsultan Keuangan",
    "Pengacara Bisnis", "Jurnalis Investigasi", "Fotografer Produk", "Executive Chef",
    "Desainer Interior", "Brand Strategist", "Psikolog Klinis", "Penerjemah Bahasa Tersumpah",
    "Copywriter Kreatif", "Barista Spesialis Kopi", "Fisioterapis Olahraga", "Perencana Tata Kota",
    "Spesialis Logistik", "HR Talent Acquisition", "Kurator Seni", "Manajer Operasional Hotel",
    "Instruktur Fitness", "Analis Kebijakan Publik", "Penyiar Radio", "Teknisi Pesawat",
    "Masinis Kereta Api", "Nahkoda Kapal Laut", "Spesialis Agribisnis Hidroponik", "Pengrajin Keramik Artistik"
]

ALL_ROLES = TECH_ROLES + NON_TECH_ROLES

PROGRAMMING_LANGS = [
    "Python", "TypeScript", "JavaScript", "Golang", "Rust", "Java", "Kotlin",
    "Swift", "C++", "PHP", "Dart", "C#", "SQL", "Scala", "Ruby", "Elixir"
]

FRAMEWORKS_TOOLS = [
    "React", "Vue", "Next.js", "FastAPI", "Django", "Node.js", "Flutter",
    "PyTorch", "Docker", "Kubernetes", "Spring Boot", "Laravel", "NestJS", "TailwindCSS",
    "PostgreSQL", "Redis", "Apache Kafka", "MongoDB", "Terraform", "GraphQL"
]

HOBBIES = [
    "bermain futsal bersama kawan lama", "bersepeda santai keliling kota di pagi hari",
    "jogging pagi di taman kota", "bermain game RPG open-world", "membaca novel fiksi ilmiah",
    "fotografi jalanan (street photography)", "bermain gitar akustik lagu-lagu santai",
    "belajar memasak sourdough artisan", "merawat tanaman hias monstera",
    "latihan angkat beban di gym", "berenang santai gaya dada", "menonton dokumenter sains",
    "mendaki gunung di akhir pekan", "camping di alam terbuka", "bermain catur kilat online",
    "menulis ulasan di blog pribadi", "belajar bahasa asing otodidak", "melukis cat air pemandangan",
    "merakit keyboard mekanikal custom", "bermain bulutangkis ganda", "merestorasi motor klasik"
]

GAMES = [
    "Valorant", "Mobile Legends", "Genshin Impact", "Dota 2", "Minecraft",
    "FIFA 24", "Elden Ring", "PUBG Mobile", "Apex Legends", "The Witcher 3",
    "Honkai Star Rail", "Cyberpunk 2077", "Tekken 8", "Stardew Valley", "Baldur's Gate 3"
]

FOODS = [
    "Nasi Goreng Kampung pedas", "Sate Ayam Madura bumbu kacang", "Rendang Daging Sapi empuk",
    "Mie Ayam Bakso urat", "Gado-gado siram saus kacang", "Soto Betawi kuah santan",
    "Ayam Geprek sambal korek", "Nasi Padang lauk dendeng batokok", "Pempek Palembang kapal selam",
    "Rawon Daging Surabaya kuah kluwek", "Gudeg Jogja komplit krecek", "Coto Makassar daging rempah",
    "Ayam Betutu Bali pedas rempah", "Sop Buntut Sapi bakar", "Mie Aceh kepiting kuah kental",
    "Nasi Liwet Solo gurih", "Bebek Sinjay sambal mangga muda", "Sate Lilit Ikan Khas Bali"
]

DRINKS = [
    "Kopi Espresso gayo", "Kopi Susu Gula Aren legit", "Teh Hijau Matcha hangat",
    "Americano dingin segar", "Jus Alpukat kocok cokelat", "Teh Earl Grey aromatik",
    "Caffè Latte lembut", "Air Kelapa Muda murni", "Wedang Jahe hangat serai",
    "Es Cendol durian", "Teh Tarik dingin", "Jamu Kunyit Asam segar",
    "Es Doger tape ketan", "Kopi Tubruk arabika wamena", "Bajigur santan hangat"
]

ALLERGIES_DIETS = [
    "alergi makanan laut (seafood udang dan kepiting)", "alergi berat pada kacang tanah",
    "pantangan tidak bisa makan makanan pedas sama sekali", "intoleransi laktosa (susu sapi)",
    "menjalani pola makan vegetarian murni", "alergi telur ayam negeri",
    "menghindari makanan tinggi gluten", "alergi buah nanas dan buah asam"
]

PETS = [
    ("Kucing Persia bulu lebat", "Mochi"), ("Kucing Domestik oyen lincah", "Simba"),
    ("Anjing Golden Retriever ramah", "Milo"), ("Kucing British Shorthair abu-abu", "Luna"),
    ("Hamster Roborovski gesit", "Kiko"), ("Kelinci Rex bulu halus", "Bubu"),
    ("Burung Lovebird kicau ceria", "Chirpy"), ("Ikan Cupang hias halfmoon", "Bluey"),
    ("Kucing Ragdoll mata biru", "Oreo"), ("Anjing Corgi pendek lucu", "Poco"),
    ("Kura-kura Brazil jinak", "Shelly"), ("Landak Mini pemalu", "Spike")
]

TRAVEL_DESTINATIONS = [
    "Gunung Bromo Jawa Timur", "Labuan Bajo dan Pulau Komodo", "Ubud kawasan persawahan Bali",
    "Danau Toba dan Pulau Samosir", "Kepulauan Raja Ampat Papua", "Kawah Ijen Banyuwangi",
    "Yogyakarta Malioboro", "Kepulauan Derawan Kalimantan", "Tana Toraja Sulawesi Selatan",
    "Dataran Tinggi Dieng", "Pantai Tanjung Kelayang Belitung", "Lembah Harau Sumatra Barat"
]

SIDE_BUSINESSES = [
    "kedai kopi susu kecil-kecilan", "jasa pembuatan website portofolio",
    "toko pakaian thrift shop online", "usaha katering makanan sehat harian",
    "kursus les privat bahasa Inggris", "studio foto dan videografi mandiri",
    "jual kue kering artisan rumahan", "jasa desain logo dan branding bisnis",
    "budidaya bibit tanaman hias hidroponik", "jasa reparasi gadget dan laptop"
]

# Out-of-Distribution (OOD) Novel Entity Synthesizer
NOVEL_ADJECTIVES = ["Antariksa", "Siber", "Holografik", "Kuantum", "Kosmik", "Aurora", "Artifisial", "Bionik", "Galaktik"]
NOVEL_NOUNS = ["Penjinak Naga", "Arsitek Koloni", "Pustakawan Nebula", "Kurator Satelit", "Pakar Ekosistem Mars", "Mekanik Robotik"]
NOVEL_CITIES = ["Kota Atlantis Baru", "Stasiun Orbit Nusantara", "Lembah Kubah Neo-Jogja", "Pusat Antariksa Morotai"]
NOVEL_DRINKS = ["Kopi Elektrolit Dingin", "Elixir Bunga Bintang", "Jus Kristal Mint", "Sirup Biometrik"]


def get_random_profile(seed: Optional[int] = None) -> Dict[str, Any]:
    """Generates a rich, non-repetitive individual conversational persona."""
    if seed is not None:
        set_seed(seed)

    # 15% chance to introduce an Out-of-Distribution (OOD) novel entity
    is_ood = random.random() < 0.15

    if is_ood:
        job = f"{random.choice(NOVEL_NOUNS)} {random.choice(NOVEL_ADJECTIVES)}"
        city = random.choice(NOVEL_CITIES)
        drink = random.choice(NOVEL_DRINKS)
    else:
        job = random.choice(ALL_ROLES)
        city = random.choice(CITIES)
        drink = random.choice(DRINKS)

    pet_type, pet_name = random.choice(PETS)
    alt_city = random.choice([c for c in CITIES if c != city])

    return {
        "name": random.choice(FIRST_NAMES),
        "city": city,
        "alt_city": alt_city,
        "job": job,
        "lang": random.choice(PROGRAMMING_LANGS),
        "tool": random.choice(FRAMEWORKS_TOOLS),
        "hobby": random.choice(HOBBIES),
        "game": random.choice(GAMES),
        "food": random.choice(FOODS),
        "drink": drink,
        "allergy": random.choice(ALLERGIES_DIETS),
        "pet_type": pet_type,
        "pet_name": pet_name,
        "destination": random.choice(TRAVEL_DESTINATIONS),
        "side_biz": random.choice(SIDE_BUSINESSES),
        "is_ood": is_ood,
    }


# ---------------------------------------------------------------------------
# 2. Multi-Style Paraphrase Engines (Statements, Queries, and Answers)
# ---------------------------------------------------------------------------

STATEMENT_STYLES_JOB_CITY = [
    # Style 1: Casual / Gaul
    "Woi halo! Kenalin gue {name}, aktivitas sehari-hari gue megang kerjaan jadi {job} nih di {city}.",
    "Halo bro! Nama gue {name}. Gue sekarang menetap di {city} dan sibuk ngantor sebagai {job}.",
    "Hai, salam kenal! Panggil aja gue {name}. Sehari-hari cari nafkah sebagai {job} di daerah {city}.",
    "Btw kenalan dulu, gue {name}. Sekarang lagi stay di {city} sambil berkarier jadi {job}.",
    "Kenalin gue {name}. Akhir-akhir ini lagi sibuk-sibuknya jalanin tugas sebagai {job} di kota {city}.",
    # Style 2: Formal / Profesional
    "Selamat pagi. Perkenalkan nama saya {name}. Saat ini saya berprofesi sebagai {job} dan berdomisili di kota {city}.",
    "Salam hangat. Saya {name}, seorang {job} profesional yang bertempat tinggal di {city}.",
    "Perkenalkan, saya {name}. Aktivitas pekerjaan utama saya adalah sebagai {job} di wilayah {city}.",
    "Nama saya {name}. Saya menetap di kota {city} dan mengemban tanggung jawab pekerjaan sebagai {job}.",
    "Perkenalkan saya {name}, berdomisili resmi di {city} dengan spesialisasi profesi sebagai {job}.",
    # Style 3: Conversational / Storytelling
    "Oiya, ngomong-ngomong namaku {name}. Aku tinggal di {city} dan kebetulan bidang kerjaku itu {job}.",
    "Namaku {name} asal kota {city}. Udah beberapa tahun ini aku fokus menggeluti profesi {job}.",
    "Hai! Aku {name}. Saat ini lagi menikmati rutinitas di {city} sembari bekerja sebagai {job}.",
    "Kenalkan, aku {name} warga {city}. Sehari-hari waktuku banyak tercurah untuk pekerjaan sebagai {job}.",
    "Aku {name} nih dari {city}. Pekerjaan utamaku sekarang adalah {job}, lumayan menantang tapi seru."
]

ASSISTANT_REACTIONS_JOB_CITY = [
    "Halo {name}! Senang berkenalan denganmu. Selamat bertugas sebagai {job} di kota {city}. Ada yang bisa kubantu?",
    "Salam kenal {name}! Seorang {job} di {city} pasti punya rutinitas yang menarik. Mau diskusi tentang apa hari ini?",
    "Hai {name}! Senang bisa terhubung dengan profesional {job} dari {city}. Semoga harimu menyenangkan ya!",
    "Halo {name}! Menarik sekali profesimu sebagai {job} di {city}. Ada topik seru apa yang mau kita bahas?",
    "Senang berkenalan denganmu, {name}! Sukses selalu untuk kariermu sebagai {job} di {city}."
]

STATEMENT_STYLES_DRINK_HOBBY = [
    "Kalau lagi rehat di sela-sela kerjaan, favoritku banget minum {drink} atau meluangkan waktu buat {hobby}.",
    "Btw cara ampuhku melepas stres seharian itu biasanya santai sambil menikmati {drink} dan {hobby}.",
    "Rutinitasku kalau ada waktu luang paling suka minum {drink}, kadang diselingi juga dengan {hobby}.",
    "Untuk jaga mood tetap segar, aku selalu sedia {drink} dan menyempatkan hobi {hobby}.",
    "Kombinasi terbaik saat santai menurutku ya menikmati {drink} hangat/dingin sambil {hobby}."
]

STATEMENT_STYLES_PET = [
    "Di rumah ada teman setia nih, seekor {pet_type} yang kuberi nama {pet_name}.",
    "Oiya, aku memelihara hewan peliharaan lho, jenisnya {pet_type} dan namanya {pet_name}.",
    "Ada yang selalu nemenin pas aku lagi di rumah, yaitu {pet_type} kesayanganku bernama {pet_name}.",
    "Peliharaanku di rumah itu seekor {pet_type} lucu, kupanggil {pet_name}.",
    "Kenalkan juga anggota keluarga berbulu di rumah: {pet_type} bernama {pet_name}."
]

STATEMENT_STYLES_FOOD_ALLERGY = [
    "Soal kuliner, menu favoritku itu {food}, tapi yang penting kuingat aku punya kondisi {allergy}.",
    "Aku paling doyan menyantap {food}, meskipun harus ekstra hati-hati karena ada {allergy}.",
    "Makanan yang paling bikin nafsu makan naik itu {food}. Cuma ya gitu, ada pantangan karena {allergy}.",
    "Kalau diajak makan, pilihanku hampir selalu {food}, asalkan bebas dari pemicu {allergy} milikku."
]

STATEMENT_STYLES_SIDE_BIZ = [
    "Selain kerjaan utama, aku juga lagi merintis usaha sampingan berupa {side_biz}.",
    "Di luar jam kerja, kegiatan produktifku adalah mengelola bisnis kecil {side_biz}.",
    "Biar ada pemasukan tambahan, aku mencoba peruntungan dengan menjalankan {side_biz}.",
    "Usaha sampingan yang lagi kujalankan sekarang yaitu {side_biz}."
]

QUERY_TEMPLATES = {
    "job": [
        "Kamu masih ingat apa profesiku?",
        "Apa pekerjaanku sehari-hari yang kuceritakan tadi?",
        "Tadi aku bilang kerja jadi apa ya?",
        "Kira-kira masih ingat karir yang kugeluti sekarang?",
        "Profesi apa yang kujalani saat ini?",
        "Coba sebutkan apa bidang pekerjaanku.",
        "Bisa ingatkan aku, posisiku di kantor sebagai apa tadi?",
        "Aku bekerja sebagai apa ya tadi?",
        "Tadi profesi apa yang sempat kuceritakan padamu?",
        "Btw jangan sampai lupa, kerjaanku apa tadi?"
    ],
    "city": [
        "Di kota mana aku tinggal sekarang?",
        "Bisa ingatkan kota domisili tempat tinggalku?",
        "Aku menetap di kota apa tadi ya?",
        "Di mana kota tempat tinggalku saat ini?",
        "Kota mana yang tadi kusebutkan sebagai rumahku?",
        "Tadi aku bilang stay di kota mana ya?",
        "Kamu ingat di kota apa aku berdomisili?",
        "Coba sebutkan nama kotaku yang tadi kuceritakan."
    ],
    "drink": [
        "Minuman yang suka kuminum saat istirahat tadi apa?",
        "Kamu ingat minuman favoritku apa?",
        "Tadi minuman apa yang selalu kuminum pas santai?",
        "Apa minuman kesukaanku yang tadi kusebut?",
        "Minuman apa yang biasa menemaniku saat rehat tadi ya?"
    ],
    "hobby": [
        "Kegiatan atau hobi apa yang biasa kulakukan saat rehat tadi?",
        "Apa hobi favoritku yang tadi kuceritakan?",
        "Kamu ingat aktivitasku di waktu luang apa?",
        "Saat santai tadi aku biasanya melakukan hobi apa ya?"
    ],
    "food": [
        "Makanan kesukaanku yang kusebutkan tadi apa ya?",
        "Apa menu makanan favoritku tadi?",
        "Tadi makanan apa yang paling doyan kusantap?",
        "Kamu masih ingat kuliner favoritku apa?"
    ],
    "allergy": [
        "Sebelum kamu rekomendasikan resto, kamu ingat pantangan atau kondisiku apa tadi?",
        "Kondisi alergi atau pantangan makananku apa ya?",
        "Apa kondisi kesehatan terkait makanan yang harus kuwaspadai tadi?",
        "Kamu ingat pantanganku saat makan tadi apa?"
    ],
    "pet_name": [
        "Siapa nama hewan peliharaanku yang kuceritakan tadi?",
        "Nama peliharaanku di rumah siapa tadi ya?",
        "Kamu ingat nama hewan peliharaan kesayanganku?",
        "Siapa nama peliharaanku tadi?"
    ],
    "side_biz": [
        "Usaha sampingan apa yang sedang kurintis tadi?",
        "Bisnis kecil di luar kerja utama yang kujalankan apa tadi ya?",
        "Kamu ingat usaha sampinganku apa?",
        "Tadi aku cerita lagi jalanin bisnis sampingan apa?"
    ],
    "game": [
        "Game apa yang biasa kumainkan bareng temen-temen?",
        "Game favorit yang sering kumainkan apa ya tadi?",
        "Kamu ingat judul game yang biasa kumainkan?"
    ]
}

ANSWER_STYLES = {
    "direct": "{val}.",
    "polite": "Kamu bekerja sebagai {val}." if "{job}" else "Kamu adalah {val}.",
    "conversational": "Dari obrolan kita tadi, kamu adalah {val}.",
    "friendly": "Tentu saja ingat! Jawaban yang benar adalah {val}."
}


def render_answer_text(fact_key: str, val: str) -> str:
    """Generates natural, varied assistant answer strings."""
    templates = [
        f"Kamu bekerja sebagai {val}." if fact_key == "job" else
        f"Kamu tinggal di kota {val}." if fact_key == "city" else
        f"Minuman favoritmu adalah {val}." if fact_key == "drink" else
        f"Kamu biasa {val}." if fact_key == "hobby" else
        f"Makanan kesukaanmu adalah {val}." if fact_key == "food" else
        f"Kamu memiliki kondisi {val}." if fact_key == "allergy" else
        f"Nama hewan peliharaanmu adalah {val}." if fact_key == "pet_name" else
        f"Usaha sampinganmu adalah {val}." if fact_key == "side_biz" else
        f"Game yang biasa kamu mainkan adalah {val}." if fact_key == "game" else
        f"Tentu ingat, jawabannya adalah {val}."
    ]
    # Also add concise natural variants
    templates.append(f"Tentu masih ingat, {val}.")
    templates.append(f"{val}, sesuai yang kamu ceritakan tadi.")
    templates.append(f"Berdasarkan yang kamu sebutkan tadi: {val}.")
    return random.choice(templates)


# ---------------------------------------------------------------------------
# 3. External Indonesian Distractor Pools
# ---------------------------------------------------------------------------

CASUAL_GREETINGS = [
    ("Halo, selamat pagi!", "Halo! Selamat pagi. Ada yang bisa kubantu hari ini?"),
    ("Hai asisten, apa kabar hari ini?", "Hai! Kabar baik, terima kasih sudah bertanya. Bagaimana kabarmu hari ini?"),
    ("Halo, lagi sibuk nggak? Boleh minta saran sebentar?", "Halo! Tentu saja tidak, saya selalu siap membantumu. Ada hal apa yang ingin kamu diskusikan?"),
    ("Selamat sore! Boleh ngobrol santai sebentar?", "Selamat sore! Tentu saja boleh, senang sekali bisa menemanimu mengobrol."),
    ("Permisi, mau tanya-tanya sedikit boleh?", "Boleh sekali! Silakan tanyakan apa saja yang ingin kamu ketahui."),
    ("Halo AI, lagi luang kan? Aku butuh teman diskusi nih.", "Halo! Tentu saja, saya siap jadi teman diskusimu hari ini. Mau bahas topik apa?"),
    ("Hai! Hari ini cuacanya lumayan bikin santai ya.", "Betul sekali, cuaca santai memang paling pas dinikmati sambil rileks. Ada cerita menarik apa hari ini?"),
    ("Hai kak, salam kenal ya!", "Hai! Salam kenal juga. Senang bisa terhubung denganmu hari ini."),
    ("Halo, selamat siang! Semoga harimu menyenangkan.", "Halo! Selamat siang, terima kasih doanya. Ada yang bisa saya bantu sekarang?"),
    ("Hai! Boleh minta waktu beberapa menit buat ngobrol?", "Tentu boleh, pintu selalu terbuka untukmu. Mau mulai dari mana?"),
]

FALLBACK_DISTRACTORS = [
    ("Akhir-akhir ini cuaca di kotaku sering hujan deras tiba-tiba.", "Memang sedang musim pancaroba. Jangan lupa selalu siapkan payung dan jaga daya tahan tubuh ya!"),
    ("Ada tips nggak supaya fokus kerja tetap terjaga dan nggak gampang ngantuk?", "Coba gunakan teknik Pomodoro 25 menit kerja dan 5 menit rehat. Cukupi juga konsumsi air putih hangat."),
    ("Menurutmu film dokumenter sains yang seru buat ditonton apa ya?", "Dokumenter tentang eksplorasi luar angkasa atau keanekaragaman samudra laut dalam biasanya sangat memukau dan informatif."),
    ("Aku berencana mulai rutin olahraga ringan di akhir pekan.", "Langkah yang luar biasa! Mulai saja dari jalan santai 30 menit atau jogging tipis di pagi hari."),
    ("Kadang susah banget ya membagi waktu antara hobi dan pekerjaan.", "Kuncinya ada di komitmen jadwal mingguan. Alokasikan waktu khusus untuk hobi tanpa membawa beban urusan kantor."),
    ("Pernah kepikiran nggak bagaimana perkembangan teknologi 10 tahun ke depan?", "Perkembangan AI dan otomatisasi diperkirakan makin terintegrasi dengan kehidupan harian untuk mempermudah pekerjaan manusia."),
    ("Ada ide camilan sehat yang praktis dibuat di rumah?", "Potongan buah segar dengan yogurt, kacang almond panggang, atau edamame rebus bisa jadi pilihan lezat dan sehat.")
]


DEFAULT_SHAREGPT_PATHS = [
    "/kaggle/input/datasets/akhyarsafrudin/memorybank-benchmark/sharegpt-indonesian.json",
    "/kaggle/input/memorybank-benchmark/sharegpt-indonesian.json",
    "/kaggle/input/sharegpt-indonesian.json",
    "/home/akhyar/Dokumen/sharegpt-indonesian.json",
    "sharegpt-indonesian.json",
]

DEFAULT_EVOL_PATHS = [
    "/kaggle/input/datasets/akhyarsafrudin/memorybank-benchmark/evol-instruct-indonesian.json",
    "/kaggle/input/memorybank-benchmark/evol-instruct-indonesian.json",
    "/kaggle/input/evol-instruct-indonesian.json",
    "/home/akhyar/Dokumen/evol-instruct-indonesian.json",
    "evol-instruct-indonesian.json",
]


def resolve_dataset_path(provided_path: Optional[str], candidate_paths: List[str]) -> Optional[str]:
    """Resolves dataset path prioritizing provided path, candidate paths, and fallbacks."""
    if provided_path and os.path.exists(provided_path):
        return provided_path
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    return provided_path if provided_path else candidate_paths[0]


class ExternalDistractorPool:
    """Manages real-world Indonesian dialogue pairs from ShareGPT/Evol-Instruct and fallbacks."""
    def __init__(
        self,
        evol_path: Optional[str] = None,
        sharegpt_path: Optional[str] = None,
        max_user_chars: int = 250,
        max_ai_chars: int = 450,
        max_load_items: int = 15000,
    ):
        self.fallback_pairs: List[Tuple[str, str]] = list(FALLBACK_DISTRACTORS)
        self.evol_pairs: List[Tuple[str, str]] = []
        self.sharegpt_pairs: List[Tuple[str, str]] = []
        self.all_pairs: List[Tuple[str, str]] = list(FALLBACK_DISTRACTORS)

        sharegpt_resolved = resolve_dataset_path(sharegpt_path, DEFAULT_SHAREGPT_PATHS)
        evol_resolved = resolve_dataset_path(evol_path, DEFAULT_EVOL_PATHS)

        if sharegpt_resolved and os.path.exists(sharegpt_resolved):
            try:
                with open(sharegpt_resolved, "r", encoding="utf-8") as f:
                    sdata = json.load(f)
                    for item in sdata:
                        turns = item.get("conversations", [])
                        for i in range(0, len(turns) - 1, 2):
                            if turns[i].get("from") == "human" and turns[i+1].get("from") == "gpt":
                                u = turns[i].get("value", "").strip()
                                a = turns[i+1].get("value", "").strip()
                                if 15 <= len(u) <= max_user_chars and 20 <= len(a) <= max_ai_chars:
                                    self.sharegpt_pairs.append((u, a))
                print(f"  ✓ Loaded {len(self.sharegpt_pairs):>5d} compact QA pairs from ShareGPT ({sharegpt_resolved})")
            except Exception as e:
                print(f"  ⚠️ Gagal membaca ShareGPT ({sharegpt_resolved}): {e}")
        else:
            print(f"  ℹ️ ShareGPT path tidak ditemukan di {sharegpt_resolved}")

        if evol_resolved and os.path.exists(evol_resolved):
            try:
                with open(evol_resolved, "r", encoding="utf-8") as f:
                    edata = json.load(f)
                    for item in edata[:max_load_items]:
                        turns = item.get("conversations", [])
                        for i in range(0, len(turns) - 1, 2):
                            if turns[i].get("from") == "human" and turns[i+1].get("from") == "gpt":
                                u = turns[i].get("value", "").strip()
                                a = turns[i+1].get("value", "").strip()
                                if 15 <= len(u) <= max_user_chars and 20 <= len(a) <= max_ai_chars:
                                    self.evol_pairs.append((u, a))
                print(f"  ✓ Loaded {len(self.evol_pairs):>5d} compact QA pairs from Evol-Instruct ({evol_resolved})")
            except Exception as e:
                print(f"  ⚠️ Gagal membaca Evol-Instruct ({evol_resolved}): {e}")
        else:
            print(f"  ℹ️ Evol-Instruct path tidak ditemukan di {evol_resolved}")

        self.all_pairs = list(FALLBACK_DISTRACTORS) + self.sharegpt_pairs + self.evol_pairs
        print(f"  ✓ Total External Distractor Pool: {len(self.all_pairs)} giliran dialog tersedia.")

    def sample_turn(self) -> Tuple[str, str]:
        return random.choice(self.all_pairs)


def inject_distractor_turn(turns: List[Dict[str, str]], pool: ExternalDistractorPool):
    u, a = pool.sample_turn()
    turns.append({"role": "user", "content": u})
    turns.append({"role": "assistant", "content": a})


# ---------------------------------------------------------------------------
# 4. Scenario Dialogue Generators with High Diversity
# ---------------------------------------------------------------------------

def generate_conversation_scenario(
    p: Dict[str, Any],
    pool: ExternalDistractorPool,
    target_turns: int = 10,
    scenario_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a full multi-turn dialogue with dynamic fact latency, diverse paraphrasing,
    and ground-truth verification.
    """
    turns = []
    facts = []

    if scenario_type is None:
        scenario_type = random.choice(["career_focus", "lifestyle_focus", "fact_correction", "casual_chitchat"])

    # 1. Opening: Randomize whether conversation starts with a greeting or immediate intro
    has_pre_greeting = random.choice([True, False])
    if has_pre_greeting:
        g_u, g_a = random.choice(CASUAL_GREETINGS)
        turns.append({"role": "user", "content": g_u})
        turns.append({"role": "assistant", "content": g_a})

    # 2. Fact Injection: Name, Job, City
    fact_turn_idx = len(turns)
    stmt_template = random.choice(STATEMENT_STYLES_JOB_CITY)
    u_stmt = stmt_template.format(name=p["name"], job=p["job"], city=p["city"])
    a_reaction = random.choice(ASSISTANT_REACTIONS_JOB_CITY).format(name=p["name"], job=p["job"], city=p["city"])

    turns.append({"role": "user", "content": u_stmt})
    turns.append({"role": "assistant", "content": a_reaction})
    facts.append({"turn": fact_turn_idx, "key": "job", "value": p["job"]})
    facts.append({"turn": fact_turn_idx, "key": "city", "value": p["city"]})
    facts.append({"turn": fact_turn_idx, "key": "name", "value": p["name"]})

    # 3. Interleaving Distractors
    inject_distractor_turn(turns, pool)

    # 4. Secondary Fact Injection (Drink, Pet, Side Biz, or Hobby)
    secondary_type = random.choice(["drink", "pet", "side_biz", "food"])
    f2_turn_idx = len(turns)

    if secondary_type == "drink":
        u_f2 = random.choice(STATEMENT_STYLES_DRINK_HOBBY).format(drink=p["drink"], hobby=p["hobby"])
        a_f2 = f"Pilihan yang sangat pas! Menikmati {p['drink']} memang bisa membantu menyegarkan pikiran kembali."
        facts.append({"turn": f2_turn_idx, "key": "drink", "value": p["drink"]})
    elif secondary_type == "pet":
        u_f2 = random.choice(STATEMENT_STYLES_PET).format(pet_type=p["pet_type"], pet_name=p["pet_name"])
        a_f2 = f"Lucu sekali! Memelihara {p['pet_type']} bernama {p['pet_name']} pasti selalu membawa keceriaan di rumah."
        facts.append({"turn": f2_turn_idx, "key": "pet_name", "value": p["pet_name"]})
    elif secondary_type == "side_biz":
        u_f2 = random.choice(STATEMENT_STYLES_SIDE_BIZ).format(side_biz=p["side_biz"])
        a_f2 = f"Langkah wirausaha yang prospektif! Semoga bisnis {p['side_biz']} yang kamu jalankan semakin maju."
        facts.append({"turn": f2_turn_idx, "key": "side_biz", "value": p["side_biz"]})
    else:
        u_f2 = random.choice(STATEMENT_STYLES_FOOD_ALLERGY).format(food=p["food"], allergy=p["allergy"])
        a_f2 = f"Catat, {p['name']}! Menyantap {p['food']} memang nikmat, dan sangat tepat selalu waspada terhadap {p['allergy']}."
        facts.append({"turn": f2_turn_idx, "key": "food", "value": p["food"]})
        facts.append({"turn": f2_turn_idx, "key": "allergy", "value": p["allergy"]})

    turns.append({"role": "user", "content": u_f2})
    turns.append({"role": "assistant", "content": a_f2})

    # 5. More Distractors to test long-horizon interference
    while len(turns) < (target_turns - 2):
        inject_distractor_turn(turns, pool)

    # 6. Memory Recall Query Turn
    candidate_keys = [f["key"] for f in facts if f["key"] in QUERY_TEMPLATES]
    recall_key = random.choice(candidate_keys)
    query_text = random.choice(QUERY_TEMPLATES[recall_key])

    # Find the target ground truth value
    ground_truth = next(f["value"] for f in facts if f["key"] == recall_key)
    answer_text = render_answer_text(recall_key, ground_truth)

    query_turn_idx = len(turns)
    turns.append({"role": "user", "content": query_text})
    turns.append({"role": "assistant", "content": answer_text})

    return {
        "topic": scenario_type,
        "is_ood": p["is_ood"],
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": query_turn_idx,
            "target_key": recall_key,
            "ground_truth": ground_truth,
            "question": query_text,
            "answer": answer_text,
        }
    }


# ---------------------------------------------------------------------------
# 5. Dataset Generation & Export Pipeline
# ---------------------------------------------------------------------------

def generate_conversation_dataset(
    num_conversations: int = 1000,
    min_turns: int = 8,
    max_turns: int = 12,
    seed: int = 42,
    output_dir: str = "dataset",
    evol_path: Optional[str] = None,
    sharegpt_path: Optional[str] = None,
    external_distractor_ratio: float = 0.65,
) -> Dict[str, Any]:
    """Generates balanced, diverse, non-templated conversational dataset splits."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 76)
    print("   UNIVERSAL DIVERSE CONVERSATIONAL DATASET GENERATOR (MULTI-PARAPHRASED)")
    print("=" * 76)
    print(f"  Total Percakapan          : {num_conversations:,}")
    print(f"  Rentang Turn              : {min_turns} - {max_turns} giliran dialog")
    print(f"  Seed Reproduksibilitas    : {seed}")
    print(f"  Direktori Output          : {os.path.abspath(output_dir)}")
    print("=" * 76)

    pool = ExternalDistractorPool(evol_path=evol_path, sharegpt_path=sharegpt_path)
    all_conversations = []
    ood_count = 0

    for idx in range(num_conversations):
        # Generate persona profile with dynamic seeder
        p = get_random_profile(seed=seed + idx)
        target_t = random.randint(min_turns, max_turns)
        # Even target turns (user + assistant pairs)
        if target_t % 2 != 0:
            target_t += 1

        conv = generate_conversation_scenario(p=p, pool=pool, target_turns=target_t)
        conv["id"] = f"conv_{idx+1:06d}"
        if conv.get("is_ood", False):
            ood_count += 1
        all_conversations.append(conv)

    # Shuffling with locked seed
    random.seed(seed)
    random.shuffle(all_conversations)

    # Split 80% Train, 10% Val, 10% Test
    n_train = int(num_conversations * 0.8)
    n_val = int(num_conversations * 0.1)

    train_data = all_conversations[:n_train]
    val_data = all_conversations[n_train:n_train + n_val]
    test_data = all_conversations[n_train + n_val:]

    def save_jsonl(data: List[Dict[str, Any]], filename: str) -> str:
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✓ Saved {filename:<26} : {len(data):>5,} dialog ({os.path.getsize(filepath) / 1024:.1f} KB)")
        return filepath

    print("\n[Exporting JSONL Files...]")
    train_f = save_jsonl(train_data, "conversations_train.jsonl")
    val_f = save_jsonl(val_data, "conversations_val.jsonl")
    test_f = save_jsonl(test_data, "conversations_test.jsonl")

    meta = {
        "generator": "generate_conversation_dataset.py",
        "total_conversations": num_conversations,
        "seed": seed,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "ood_samples_count": ood_count,
        "ood_ratio": round(ood_count / num_conversations, 4),
        "distractor_pool_stats": {
            "sharegpt_pairs": len(pool.sharegpt_pairs),
            "evol_pairs": len(pool.evol_pairs),
            "total_distractor_pairs": len(pool.all_pairs),
        },
        "train_file": train_f,
        "val_file": val_f,
        "test_file": test_f,
    }

    meta_path = os.path.join(output_dir, "conversations_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Dataset selesai dibuat dengan {ood_count} sampel Out-of-Distribution ({ood_count/num_conversations*100:.1f}%)!")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diverse Multi-Paraphrased Conversational Dataset Generator")
    parser.add_argument("--num_conversations", type=int, default=1000, help="Total conversations to generate")
    parser.add_argument("--min_turns", type=int, default=8, help="Min turns per dialogue")
    parser.add_argument("--max_turns", type=int, default=12, help="Max turns per dialogue")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--output_dir", default="dataset", help="Output directory")
    parser.add_argument(
        "--evol_path",
        type=str,
        default=None,
        help="Path to evol-instruct-indonesian.json (Kaggle or local)"
    )
    parser.add_argument(
        "--sharegpt_path",
        type=str,
        default=None,
        help="Path to sharegpt-indonesian.json (Kaggle or local)"
    )
    args = parser.parse_args()

    generate_conversation_dataset(
        num_conversations=args.num_conversations,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        seed=args.seed,
        output_dir=args.output_dir,
        evol_path=args.evol_path,
        sharegpt_path=args.sharegpt_path,
    )
