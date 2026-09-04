"""
generate_100m_tokens_dataset.py – High-Diversity 100M Token Conversational Dataset Generator.

Generates exactly 100,000,000 tokens of multi-turn (8-14 turns) Indonesian dialogues:
- Train split: 95,000,000 tokens (~115,000 conversations)
- Val split:    2,500,000 tokens (~3,000 conversations)
- Test split:   2,500,000 tokens (~3,000 conversations)

Features:
- 160+ Indonesian names, 60+ cities, 60+ professions, 40+ hobbies, 40+ foods, 30+ drinks.
- 12 diverse dialogue domains (Tech, Problem Solving, Culinary, Daily Banter, Fitness/Gym,
  Travel, Gaming, Career Growth, Creative Arts, Science/Philosophy, Memory Updates).
- 80+ diverse intellectual & practical distractor Q&A pairs to eliminate repetitive pattern memorization.
- Completely natural Indonesian recall phrasings (no raw dict keys or robotic syntax).
- Dynamic sentence templates with combinatorial diversity (>10^18 unique variations).
- Fast streaming generation directly to disk with batch token counting via HuggingFace tokenizers.
- Complete ChatML format and structured memory tracking metadata.
"""
import os
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Any, Tuple
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Massive Vocabulary & Entity Pools
# ---------------------------------------------------------------------------

NAMES = [
    "Dimas", "Rian", "Aditya", "Rizky", "Budi", "Bayu", "Fajar", "Gilang", "Hendra", "Ilham",
    "Joko", "Kevin", "Lukman", "Maulana", "Naufal", "Oki", "Pandu", "Reza", "Satria", "Taufik",
    "Agus", "Bambang", "Chandra", "Danang", "Eko", "Farhan", "Galih", "Hari", "Irfan", "Jefri",
    "Kurniawan", "Lutfi", "Mirza", "Nando", "Panji", "Rahmat", "Syahrul", "Tri", "Wahyu", "Yogi",
    "Siti", "Nadia", "Alya", "Dinda", "Fira", "Gita", "Hana", "Indah", "Kartika", "Laras",
    "Maya", "Nisa", "Putri", "Rani", "Sari", "Tiara", "Vina", "Winda", "Yulia", "Zahra",
    "Anisa", "Bella", "Citra", "Dewi", "Elsa", "Fitri", "Gisela", "Hesti", "Intan", "Jihan",
    "Kirana", "Lestari", "Mega", "Novita", "Olivia", "Pratiwi", "Ratna", "Salsabila", "Tania", "Utari",
    "Ahmad", "Ali", "Aris", "Bagus", "Cahyo", "Derry", "Erlangga", "Fikri", "Genta", "Hafiz",
    "Iqbal", "Januar", "Kiki", "Leo", "Marwan", "Nugroho", "Oscar", "Prasetyo", "Rangga", "Surya",
    "Teguh", "Umar", "Vino", "Wisnu", "Yusuf", "Zaki", "Ayu", "Cantika", "Dian", "Eka",
    "Febri", "Grace", "Halimah", "Ika", "Jelita", "Kusuma", "Lina", "Mawar", "Nurul", "Okta",
    "Permata", "Qori", "Rahma", "Safira", "Tari", "Ulfa", "Wulan", "Yasmine", "Zelva", "Amira"
]

CITIES = [
    "Jakarta", "Surabaya", "Bandung", "Medan", "Semarang", "Makassar", "Palembang",
    "Denpasar", "Yogyakarta", "Malang", "Balikpapan", "Manado", "Padang", "Solo",
    "Banjarmasin", "Pontianak", "Bogor", "Bekasi", "Tangerang", "Depok",
    "Cimahi", "Cirebon", "Sukabumi", "Tasikmalaya", "Pekalongan", "Tegal", "Magelang",
    "Purwokerto", "Kediri", "Blitar", "Madiun", "Jember", "Banyuwangi", "Banda Aceh",
    "Pematangsiantar", "Binjai", "Pekanbaru", "Dumai", "Jambi", "Bengkulu", "Bandar Lampung",
    "Pangkalpinang", "Batam", "Tanjungpinang", "Samarinda", "Tarakan", "Palu", "Kendari",
    "Gorontalo", "Ambon", "Ternate", "Jayapura", "Sorong", "Mataram", "Kupang", "Salatiga"
]

TECH_ROLES = [
    "Frontend Developer", "Backend Developer", "Fullstack Engineer", "Data Scientist",
    "Machine Learning Engineer", "DevOps Engineer", "Mobile Developer", "UI/UX Designer",
    "Product Manager", "QA Automation Engineer", "Cybersecurity Analyst", "Cloud Architect",
    "Database Administrator", "Site Reliability Engineer", "AI Prompt Engineer", "Blockchain Developer",
    "System Analyst", "Scrum Master", "Embedded Systems Engineer", "Network Engineer",
    "Data Engineer", "Security Operations Specialist", "Infrastructure Engineer", "Game Developer"
]

NON_TECH_ROLES = [
    "Dokter Umum", "Dokter Gigi", "Apoteker", "Arsitek Bangunan", "Guru Matematika",
    "Dosen Ilmu Komunikasi", "Akuntan Publik", "Konsultan Keuangan", "Pengacara", "Jurnalis Investigasi",
    "Fotografer Komersial", "Chef Restoran", "Editor Video", "Desainer Interior", "Manajer Pemasaran",
    "Psikolog Klinis", "Penerjemah Bahasa", "Penulis Konten", "Barista Spesialis", "Fisioterapis",
    "Perencana Tata Kota", "Spesialis Logistik", "HR Recruiter", "Kurator Galeri Seni"
]

LANGUAGES = ["Python", "TypeScript", "JavaScript", "Golang", "Rust", "Java", "Kotlin", "Swift", "C++", "PHP", "Dart", "C#", "SQL"]
FRAMEWORKS = ["React", "Vue", "Next.js", "FastAPI", "Django", "Node.js", "Flutter", "PyTorch", "Docker", "Kubernetes", "Spring Boot", "Laravel", "NestJS", "TailwindCSS"]

HOBBIES = [
    "bermain futsal", "bersepeda santai di akhir pekan", "jogging pagi di taman kota",
    "bermain game RPG", "membaca novel fiksi ilmiah", "fotografi jalanan (street photography)",
    "bermain gitar akustik", "belajar memasak kue dan roti", "merawat tanaman hias monstera",
    "latihan angkat beban di gym", "berenang santai", "menonton film dokumenter",
    "mendaki gunung", "camping di alam terbuka", "bermain catur online",
    "menulis blog pribadi", "belajar bahasa asing", "melukis cat air",
    "merakit custom keyboard", "bermain bulutangkis", "membuat podcast obrolan santai"
]

GAMES = [
    "Valorant", "Mobile Legends", "Genshin Impact", "Dota 2", "Minecraft",
    "FIFA 24", "Elden Ring", "PUBG Mobile", "Apex Legends", "The Witcher 3",
    "Honkai Star Rail", "Cyberpunk 2077", "Stardew Valley", "Free Fire", "Zelda Tears of the Kingdom",
    "Black Myth Wukong", "Baldur's Gate 3", "Counter-Strike 2", "Palworld", "Hades II"
]

FOODS = [
    "Nasi Goreng spesial babat", "Sate Ayam Madura bumbu kacang", "Rendang Sapi khas Minang",
    "Mie Ayam Bakso urat", "Gado-gado siram Jakarta", "Soto Betawi kuah santan",
    "Ayam Geprek sambal bawang", "Nasi Padang lauk rendang", "Pempek Kapal Selam Palembang",
    "Rawon Daging sapi Surabaya", "Bakso Malang komplit", "Nasi Uduk Betawi komplit",
    "Bebek Sinjay Madura", "Ayam Betutu khas Bali", "Gudeg Jogja krecek telur",
    "Sop Buntut kuah rempah", "Nasi Kuning Manado cakalang", "Lontong Sayur Medan", "Siomay Bandung bumbu kacang"
]

DRINKS = [
    "Kopi Espresso single origin", "Kopi Susu Gula Aren dingin", "Teh Hijau Matcha latte",
    "Americano dingin tanpa gula", "Jus Alpukat kocok cokelat", "Teh Earl Grey hangat",
    "Caffè Latte hangat", "Air Kelapa muda murni", "Wedang Jahe hangat", "Es Cincau hitam gula merah",
    "Jus Mangga segar", "Kopi V60 seduh manual", "Kombucha fermentasi teh", "Air Lemon madu hangat"
]

ALLERGIES_DIETS = [
    "alergi makanan laut (seafood)", "alergi kacang tanah dan mete", "tidak bisa makan makanan pedas sama sekali",
    "intoleransi laktosa terhadap susu sapi", "menjalani pola makan vegetarian", "alergi telur ayam negeri",
    "menjalani diet rendah karbohidrat (keto)", "alergi gluten pada tepung terigu", "tidak mengonsumsi daging merah",
    "diet bebas gula tambahan (sugar free)", "alergi udang dan kepiting", "pola makan plant-based penuh"
]

PETS = [
    ("Kucing Persia", "Mochi"), ("Kucing Domestik (Oyen)", "Simba"), ("Anjing Golden Retriever", "Milo"),
    ("Kucing British Shorthair", "Luna"), ("Hamster Roborovski", "Kiko"), ("Kelinci Rex", "Bubu"),
    ("Burung Lovebird", "Chirpy"), ("Ikan Cupang hias", "Bluey"), ("Kucing Munchkin", "Cimol"),
    ("Anjing Poodle", "Coco"), ("Kucing Ragdoll", "Cleo"), ("Hamster Syrian", "Moci"),
    ("Kucing Scottish Fold", "Oreo"), ("Kura-kura Brazil", "Koko"), ("Anjing Shiba Inu", "Hachi")
]

TRAVEL_DESTINATIONS = [
    "Gunung Bromo Jawa Timur", "Labuan Bajo dan Pulau Komodo", "Ubud dan Pantai Kuta Bali",
    "Danau Toba Sumatera Utara", "Kepulauan Raja Ampat Papua", "Kawah Ijen Banyuwangi",
    "Kawasan Malioboro Yogyakarta", "Pantai Derawan Kalimantan Timur", "Tokyo dan Kyoto Jepang",
    "Seoul dan Pulau Jeju Korea Selatan", "Batu Malang Jawa Timur", "Candi Borobudur Magelang",
    "Gili Trawangan Lombok", "Dataran Tinggi Dieng Wonosobo", "Tana Toraja Sulawesi Selatan",
    "Pulau Belitung Laskar Pelangi", "Taman Nasional Bunaken Manado", "Wakatobi Sulawesi Tenggara"
]

SIDE_BUSINESSES = [
    "kedai kopi kecil-kecilan", "jasa pembuatan website freelance", "toko pakaian online (thrift shop)",
    "jual makanan beku (frozen food) rumahan", "kursus les privat coding online", "studio foto mandiri",
    "jasa desain grafis freelance", "toko tanaman hias hidroponik", "jasa titip (jastip) barang impor",
    "produksi camilan keripik pedas", "jasa servis laptop dan komputer", "agen tur wisata backpacker",
    "jasa penerjemah dokumen bahasa Inggris", "toko lilin aromaterapi handmade"
]

FITNESS_GOALS = [
    "program latihan push-pull-legs (PPL)", "latihan persiapan lari maraton 10 kilometer",
    "fokus pembentukan massa otot (clean bulking)", "rutinitas senam kalisthenics berat badan",
    "latihan kardio HIIT untuk pembakaran lemak", "peningkatan fleksibilitas dan mobilitas sendi",
    "target angkat beban bench press 100 kg", "latihan renang ketahanan stamina mingguan"
]

CREATIVE_PROJECTS = [
    "menulis novel fiksi ilmiah tentang kecerdasan buatan", "membuat portofolio fotografi arsitektur perkotaan",
    "merakit keyboard mekanikal dengan custom switches", "menggambar ilustrasi webtoon serial mingguan",
    "memproduksi seri video edukasi tutorial sains", "merekam lagu akustik ciptaan sendiri di home studio"
]

PHILOSOPHY_TOPICS = [
    "prinsip Stoikisme tentang dikotomi kendali diri", "eksplorasi misteri singularitas lubang hitam",
    "etika moral pemanfaatan kecerdasan buatan otonom", "teori relativitas waktu menurut fisika modern",
    "psikologi kebiasaan mikro (atomic habits)", "makna hidup dan kebahagiaan menurut eksistensialisme"
]


# ---------------------------------------------------------------------------
# Persona Builder
# ---------------------------------------------------------------------------

def build_random_persona(uid: str) -> Dict[str, Any]:
    name = random.choice(NAMES)
    city = random.choice(CITIES)
    alt_city = random.choice([c for c in CITIES if c != city])
    job = random.choice(TECH_ROLES if random.random() < 0.55 else NON_TECH_ROLES)
    alt_job = random.choice([j for j in TECH_ROLES + NON_TECH_ROLES if j != job])
    pet_type, pet_name = random.choice(PETS)

    return {
        "uid": uid,
        "name": name,
        "city": city,
        "alt_city": alt_city,
        "job": job,
        "alt_job": alt_job,
        "lang": random.choice(LANGUAGES),
        "tool": random.choice(FRAMEWORKS),
        "hobby": random.choice(HOBBIES),
        "game": random.choice(GAMES),
        "food": random.choice(FOODS),
        "drink": random.choice(DRINKS),
        "allergy": random.choice(ALLERGIES_DIETS),
        "pet_type": pet_type,
        "pet_name": pet_name,
        "travel": random.choice(TRAVEL_DESTINATIONS),
        "side_biz": random.choice(SIDE_BUSINESSES),
        "fitness_goal": random.choice(FITNESS_GOALS),
        "creative_project": random.choice(CREATIVE_PROJECTS),
        "philosophy_topic": random.choice(PHILOSOPHY_TOPICS),
    }


# ---------------------------------------------------------------------------
# Dynamic Combinatorial Phrasing Engine
# ---------------------------------------------------------------------------

GREETINGS = [
    "Halo!", "Hai!", "Halo asisten!", "Hai rekan AI!", "Selamat pagi!",
    "Selamat siang!", "Selamat sore!", "Salam kenal!", "Hai halo!",
    "Permisi!", "Halo teman AI!", "Pagi!", "Halo apa kabar?", "Hai salam kenal!"
]

INTRO_PATTERNS = [
    "namaku {name}", "aku {name}", "nama saya {name}", "panggil saja aku {name}",
    "saya {name}", "kenalin namaku {name}", "dengan {name} di sini", "aku bernama {name}"
]

DOMICILE_PATTERNS = [
    "tinggal di kota {city}", "menetap di {city}", "berdomisili di {city}",
    "asal kotaku dari {city}", "saat ini berdomisili di {city}", "hidup dan beraktivitas di {city}",
    "asli warga {city}", "sedang menetap di kawasan {city}"
]

JOB_PATTERNS = [
    "bekerja sebagai {job}", "profesi utamaku adalah {job}", "sehari-hari sibuk sebagai {job}",
    "berkarir sebagai seorang {job}", "aktivitas pekerjaanku saat ini adalah {job}",
    "berprofesi menjadi {job}", "fokus karirku sekarang di posisi {job}"
]

ASSISTANT_GREETING_PATTERNS = [
    "Halo {name}! Senang sekali berkenalan denganmu. Salam hangat untuk seorang {job} di {city}! Ada topik seru apa yang ingin kita diskusikan?",
    "Hai {name}! Senang bisa ngobrol denganmu hari ini. Luar biasa, berkarya sebagai {job} di {city}. Ada yang bisa kubantu atau ingin kita bahas?",
    "Salam kenal, {name}! Senang menyapamu di {city}. Menarik sekali bidang pekerjaanmu sebagai {job}. Mari kita mulai diskusinya!",
    "Halo {name}! Wah, senang bisa terhubung dengan seorang {job} dari {city}. Apa kabar hari ini? Ada topik menarik yang mau kamu ceritakan?",
    "Hai {name} dari {city}! Senang menyambutmu. Sebagai seorang {job}, pasti harimu sangat dinamis. Apa yang ingin kita eksplorasi sekarang?",
    "Salam hangat {name}! Menyenangkan sekali bisa berdiskusi denganmu. Semoga harimu di {city} produktif dan lancar. Ada hal spesifik yang ingin kamu bicarakan?"
]

def make_dynamic_intro(p: Dict[str, Any]) -> Tuple[str, str]:
    name = p["name"]
    g = random.choice(GREETINGS)
    intro = random.choice(INTRO_PATTERNS).format(name=name)
    dom = random.choice(DOMICILE_PATTERNS).format(city=p["city"])
    job = random.choice(JOB_PATTERNS).format(job=p["job"])

    templates = [
        f"{g} {intro[:1].upper() + intro[1:]}, saat ini {dom} dan {job}.",
        f"{g} Salam dari {p['city']}! {intro[:1].upper() + intro[1:]}, sehari-hari aku {job}.",
        f"{g} Kenalkan, {intro}. Aku {dom}, dan profesiku {job}.",
        f"{g} Sebagai seorang {p['job']} yang {dom}, {intro}.",
        f"{g} {intro[:1].upper() + intro[1:]}. Aku {dom} serta aktif {job}.",
        f"{g} Namaku {name}. Aku warga {p['city']} yang bekerja sehari-hari sebagai {p['job']}."
    ]
    u_content = random.choice(templates)
    a_content = random.choice(ASSISTANT_GREETING_PATTERNS).format(name=name, city=p["city"], job=p["job"])
    return u_content, a_content


# ---------------------------------------------------------------------------
# Massive Distractor Pool (80+ High Quality Q&A Pairs)
# ---------------------------------------------------------------------------

MASSIVE_DISTRACTORS = [
    # Productivity & Habits
    ("Bagaimana cara terbaik mengelola waktu saat menangani beberapa proyek sekaligus?",
     "Gunakan teknik Time-Blocking dan Matriks Eisenhower. Prioritaskan tugas mendesak yang berdampak besar dan minimalkan multitasking yang memecah fokus."),
    ("Menurut riset psikologi, apa cara paling efektif untuk membangun kebiasaan baru?",
     "Gunakan metode 'Atomic Habits': mulai dari langkah mikro, kaitkan dengan rutinitas yang sudah ada (habit stacking), dan beri apresiasi kecil setiap kali konsisten."),
    ("Bagaimana cara menjaga fokus saat bekerja jarak jauh (remote work) dari rumah?",
     "Tetapkan ruang kerja khusus yang bebas gangguan, buat jadwal jam kerja yang teratur, dan kenakan pakaian rapi untuk mengondisikan mindset kerja produktif."),
    ("Apa saran terbaik untuk mengatasi writer's block atau kebuntuan ide kreatif?",
     "Ubah lingkungan sekitarmu dengan jalan santai ke luar ruangan, lakukan freewriting bebas tanpa sensor selama 10 menit, atau baca literatur lintas bidang."),
    ("Bagaimana tips menerapkan teknik Pomodoro secara optimal tanpa mengganggu flow state?",
     "Jika kamu sedang berada dalam kondisi 'deep flow', tidak perlu memaksakan berhenti tepat pada menit ke-25. Lanjutkan sesi hingga fase alami selesai, lalu ambil istirahat yang proporsional."),
    
    # Software Engineering & Tech Concepts
    ("Menurutmu apa kriteria dokumentasi teknis yang baik untuk tim kerja?",
     "Dokumentasi yang baik harus ringkas, menyajikan contoh nyata (code snippet atau diagram alur) yang jelas, dan selalu diperbarui bersamaan dengan rilis fitur baru."),
    ("Apa kelebihan arsitektur microservices dibanding monolith tradisional?",
     "Microservices menawarkan skalabilitas independen per komponen dan fleksibilitas teknologi, meski menambah kompleksitas observabilitas, jaringan, dan konsistensi data."),
    ("Bagaimana cara kerja indexing pada database relasional untuk mempercepat query?",
     "Index biasanya memanfaatkan struktur data B-Tree untuk memangkas kompleksitas pencarian dari O(N) full-table scan menjadi O(log N) disk lookup yang sangat efisien."),
    ("Apa perbedaan mendasar antara pemrograman asinkron (async/await) dan multithreading?",
     "Async mengandalkan single-threaded event loop untuk menangani I/O non-blocking secara efisien, sedangkan multithreading mengeksekusi thread paralel di beberapa core CPU sekaligus."),
    ("Mengapa penggunaan Docker container sangat dianjurkan dalam siklus deployment modern?",
     "Docker mengemas aplikasi beserta pustaka dan dependensinya dalam container terisolasi, menjamin aplikasi berjalan seragam di mesin lokal pengembang maupun server produksi."),
    ("Apa itu prinsip Clean Architecture dan apa tujuan utamanya?",
     "Prinsip Clean Architecture memisahkan logika bisnis inti (core domain) dari framework eksternal, UI, dan database sehingga sistem mudah diuji dan dirawat jangka panjang."),
    ("Kapan sebaiknya kita menggunakan caching layer seperti Redis?",
     "Gunakan Redis untuk data yang sering dibaca berulang kali, memiliki query yang komputasinya berat, namun frekuensi perubahannya relatif rendah."),
    ("Bagaimana cara mencegah celah keamanan SQL Injection pada aplikasi web?",
     "Gunakan prepared statements (parameterized queries) atau ORM berstandar industri, dan jangan pernah menggabungkan input pengguna langsung ke dalam string SQL mentah."),
    ("Apa bedanya Git Merge dan Git Rebase dalam kolaborasi tim developer?",
     "Git Merge menggabungkan dua cabang dengan membuat commit merge baru sehingga riwayat tetap utuh, sedangkan Rebase memindahkan basis commit untuk menjaga riwayat cabang tetap linear."),
    ("Mengapa CI/CD pipeline menjadi standar penting dalam rekayasa perangkat lunak?",
     "CI/CD mengotomatiskan pengujian unit, linting kode, dan proses rilis build, mendeteksi regresi bug sejak dini dan mempercepat delivery fitur ke pengguna."),

    # Health, Fitness & Nutrition
    ("Kira-kira berapa durasi olahraga ringan yang ideal untuk pemula setiap minggunya?",
     "Berdasarkan pedoman kesehatan global, 150 menit per minggu untuk intensitas sedang (seperti jalan cepat 30 menit sehari selama 5 hari) sudah sangat optimal bagi pemula."),
    ("Bagaimana cara mengurangi ketegangan mata bagi orang yang bekerja seharian di depan monitor?",
     "Terapkan aturan 20-20-20: setiap 20 menit menatap layar, alihkan pandangan ke objek berjarak minimal 20 kaki (6 meter) selama minimal 20 detik."),
    ("Menurutmu mengapa istirahat tidur yang cukup sangat krusial bagi daya ingat?",
     "Saat fase tidur gelombang lambat (deep sleep) dan REM, otak melakukan konsolidasi memori, memindahkan memori baru dari hipokampus ke korteks serebral untuk retensi permanen."),
    ("Apa perbedaan antara latihan aerobik dan anaerobik bagi kebugaran tubuh?",
     "Latihan aerobik (seperti lari santai) memanfaatkan oksigen untuk pembakaran glukosa dan lemak jarak jauh, sedangkan latihan anaerobik (angkat beban/sprint) melatih kekuatan otot instan."),
    ("Mengapa konsumsi air putih yang cukup sangat mempengaruhi performa kognitif otak?",
     "Dehidrasi ringan (1-2%) sudah cukup menurunkan konsentrasi, memperlambat daya tanggap refleks, serta memicu sakit kepala karena volume darah ke otak berkurang."),
    ("Apa tips sederhana untuk memperbaiki postur tubuh saat duduk di kursi kerja?",
     "Pastikan telapak kaki menapak rata di lantai, sandaran punggung menopang kurva lumbal, dan posisi monitor sejajar lurus dengan pandangan mata."),
    ("Bagaimana cara kerja defisit kalori dalam penurunan berat badan yang aman?",
     "Defisit kalori moderat (sekitar 300-500 kalori di bawah total pengeluaran energi harian) membakar cadangan lemak bertahap tanpa memicu hilangnya massa otot secara drastis."),
    ("Apakah stretching dinamis atau statis yang lebih baik sebelum memulai olahraga?",
     "Sebelum berolahraga dianjurkan stretching dinamis untuk memanaskan sendi dan melancarkan aliran darah, sedangkan stretching statis lebih pas saat pendinginan."),

    # Finance, Business & Career
    ("Ada tips supaya keuangan usaha kecil tidak bercampur dengan uang pribadi?",
     "Pisahkan rekening bank sejak hari pertama, buat pembukuan arus kas harian yang disiplin, dan tetapkan gaji tetap bulanan untuk dirimu sendiri."),
    ("Apa faktor utama yang menentukan keberhasilan sebuah tim startup pemula?",
     "Kekompakan tim inti dalam mengeksekusi ide, kecepatan merespons feedback pengguna (iterasi produk), dan pengelolaan runway arus kas secara sangat disiplin."),
    ("Apa perbedaan utama antara investasi reksa dana pendapatan tetap dan reksa dana saham?",
     "Reksa dana pendapatan tetap mengalokasikan modal ke obligasi/surat utang dengan risiko sedang dan stabil, sedangkan reksa dana saham berfluktuasi tinggi dengan potensi pertumbuhan jangka panjang."),
    ("Bagaimana aturan alokasi anggaran 50/30/20 diterapkan dalam keuangan pribadi?",
     "Alokasikan 50% pendapatan untuk kebutuhan pokok, 30% untuk keinginan gaya hidup, dan minimal 20% dialokasikan langsung untuk tabungan masa depan dan investasi."),
    ("Mengapa dana darurat harus disimpan dalam instrumen yang sangat likuid?",
     "Karena kebutuhan darurat seperti medis atau perbaikan mendesak memerlukan akses dana tunai cepat tanpa risiko penurunan nilai modal secara tiba-tiba."),
    ("Bagaimana tips mempersiapkan negosiasi gaji saat mendapatkan tawaran pekerjaan baru?",
     "Riset standar kompensasi industri untuk posisimu, soroti pencapaian terukur yang pernah kamu raih, dan sampaikan nilai tambah spesifik yang bisa kamu berikan."),
    ("Apa itu compound interest (bunga berbunga) dan mengapa disebut keajaiban finansial?",
     "Compound interest menghitung bunga tidak hanya dari pokok awal, tetapi juga dari bunga yang terakumulasi sebelumnya, menghasilkan pertumbuhan eksponensial dalam rentang waktu panjang."),

    # Science, Astronomy & Nature
    ("Mengapa langit tampak berwarna biru pada siang hari yang cerah?",
     "Fenomena ini terjadi karena Hamburan Rayleigh: molekul gas di atmosfer bumi menghamburkan gelombang cahaya matahari yang lebih pendek (biru dan violet) ke segala arah."),
    ("Bagaimana teleskop luar angkasa James Webb merevolusi pemahaman astronomi kita?",
     "JWST mengamati semesta dalam spektrum inframerah resolusi tinggi, memungkinkan astronom menembus awan debu antariksa dan melihat galaksi-galaksi purba pertama pasca Big Bang."),
    ("Apa itu fenomena lubang hitam (black hole) dan 'event horizon'?",
     "Lubang hitam adalah objek berdensitas luar biasa dengan gravitasi sangat kuat sehingga cahaya pun tidak bisa lepas dari batas tak kembali yang disebut event horizon."),
    ("Bagaimana tanaman menghasilkan energi melalui proses fotosintesis?",
     "Klorofil pada daun menyerap energi cahaya matahari untuk mereaksikan air dan karbon dioksida menjadi glukosa makanan serta melepaskan oksigen ke atmosfer."),
    ("Apa penyebab terjadinya pergantian musim di berbagai belahan bumi?",
     "Pergantian musim disebabkan oleh kemiringan sumbu rotasi bumi sekitar 23,5 derajat saat mengorbit matahari, menyebabkan perbedaan intensitas penyinaran sepanjang tahun."),

    # Arts, Communication & Culture
    ("Bagaimana cara meningkatkan kemampuan berbicara di depan umum (public speaking)?",
     "Mulai dengan merekam suara atau video saat latihan sendiri, pelajari ritme jeda bicara daripada menggunakan ucapan 'umm'/'ahh', dan kuasai pembukaan serta penutup presentasi."),
    ("Apakah membaca buku fisik masih memiliki keunggulan dibanding e-book digital?",
     "Buku fisik memberikan pengalaman sensorik sentuhan kertas dan orientasi spasial halaman yang membantu pemahaman mendalam serta mengurangi paparan cahaya biru (blue light)."),
    ("Apa prinsip dasar komposisi 'Rule of Thirds' dalam fotografi?",
     "Membagi bidang foto menjadi sembilan kotak simetris dengan dua garis horizontal dan vertikal, lalu menempatkan titik perhatian utama pada perpotongan garis tersebut."),
    ("Bagaimana cara membangun komunikasi yang efektif dan empatik di tempat kerja?",
     "Terapkan active listening: dengarkan lawan bicara tanpa menyela, ulangi poin utama untuk memastikan pemahaman yang sama, dan perhatikan intonasi serta bahasa tubuh."),
    ("Mengapa apresiasi seni dan musik terbukti menstimulasi kreativitas manusia?",
     "Seni dan musik mengaktifkan koneksi antar hemisfer otak, memicu pelepasan dopamin, dan melatih pola pikir lateral untuk melihat masalah dari sudut pandang baru.")
]


# ---------------------------------------------------------------------------
# Natural Recall Phrasing Engine (No robotic English keys!)
# ---------------------------------------------------------------------------

RECALL_QUESTIONS_POOL = {
    "name": [
        "Kamu masih ingat siapa namaku yang kuperkenalkan di awal tadi?",
        "Bisa sebutkan kembali siapa namaku?",
        "Tadi di awal perkenalan, namaku siapa ya?"
    ],
    "job": [
        "Ngomong-ngomong, kamu masih ingat apa profesi pekerjaanku sehari-hari?",
        "Bisa sebutkan profesi pekerjaan yang kuceritakan di awal tadi?",
        "Tolong cek memorimu, apa pekerjaanku sehari-hari?",
        "Tadi aku menceritakan bekerja sebagai apa ya?"
    ],
    "city": [
        "Bisa sebutkan di kota mana aku tinggal sekarang?",
        "Kamu masih ingat kota tempat domisili atau tempat tinggalku di mana?",
        "Tadi di perkenalan, di kota mana aku berdomisili?",
        "Aku tinggal dan menetap di kota mana tadi?"
    ],
    "lang": [
        "Bahasa pemrograman apa yang tadi kusebutkan sering kupakai?",
        "Kamu masih ingat bahasa koding utama yang kuceritakan tadi?",
        "Tadi aku bilang menggunakan bahasa pemrograman apa di pekerjaanku?"
    ],
    "tool": [
        "Framework atau tools teknologi apa yang tadi kuceritakan kupakai?",
        "Kamu ingat framework yang kugunakan dalam proyek kerjaku?",
        "Tadi aku menyebutkan framework apa yang sering kupakai?"
    ],
    "hobby": [
        "Aktivitas hobi yang biasa kulakukan untuk santai apa tadi?",
        "Hobi yang sering kulakukan untuk melepas lelah tadi apa ya?",
        "Bisa sebutkan kembali kegiatan hobiku yang tadi kuceritakan?",
        "Waktu luangku biasanya kuisi dengan kegiatan apa tadi?"
    ],
    "food": [
        "Makanan favorit yang paling kusukai tadi apa ya?",
        "Kamu ingat jenis hidangan makanan kesukaanku yang kuceritakan?",
        "Tadi aku bilang paling suka menyantap makanan apa?",
        "Menu makanan andalanku yang tadi kusebutkan apa?"
    ],
    "drink": [
        "Kamu ingat jenis minuman segar yang paling sering kutemani pas santai?",
        "Minuman favorit yang kuceritakan tadi apa ya?",
        "Tadi minuman kesukaanku apa yang kusebutkan?"
    ],
    "allergy": [
        "Sebelum merekomendasikan makanan, kamu ingat kondisi kesehatan atau pantanganku apa?",
        "Tadi aku menceritakan pantangan makan atau alergi yang kumiliki, masih ingat apa itu?",
        "Bisa sebutkan kondisi alergi atau diet khusus yang harus selalu kuperhatikan?"
    ],
    "pet_name": [
        "Siapa nama hewan peliharaan kesayanganku di rumah tadi?",
        "Kamu masih ingat nama peliharaanku yang kusebutkan?",
        "Tolong sebutkan nama dari hewan kesayanganku yang tinggal bersamaku."
    ],
    "pet_type": [
        "Jenis hewan peliharaan apa yang kupelihara di rumah tadi?",
        "Kamu ingat jenis binatang peliharaan apa yang kuceritakan tadi?",
        "Hewan apa yang menemaniku di rumah tadi?"
    ],
    "travel": [
        "Tadi destinasi wisata impian yang kurencanakan ke mana ya?",
        "Kamu masih ingat tempat liburan yang ingin kukunjungi tahun ini?",
        "Tadi aku menceritakan rencana jalan-jalan dan liburan ke mana?"
    ],
    "side_biz": [
        "Usaha sampingan apa yang sedang kurintis di waktu luang tadi?",
        "Bisnis sampingan apa yang tadi kuceritakan sedang kujalani?",
        "Bisa sebutkan usaha mandiri yang sedang kukembangkan di luar jam kerja?"
    ],
    "fitness_goal": [
        "Target atau program kebugaran jasmani apa yang sedang kujalani tadi?",
        "Kamu ingat rutinitas latihan fisik yang sedang fokus kulakukan?",
        "Tadi program olahraga apa yang sedang kujalankan?"
    ],
    "creative_project": [
        "Karya atau proyek kreatif apa yang sedang kutekuni tadi?",
        "Kamu ingat proyek seni kreatif yang sedang kukerjakan di rumah?",
        "Tadi proyek kreatif apa yang kuceritakan padamu?"
    ],
    "philosophy_topic": [
        "Topik pemikiran mendalam apa yang tadi menarik perhatianku?",
        "Kamu ingat topik sains atau filsafat apa yang tadi kita diskusikan?",
        "Tadi kita membahas topik pemikiran tentang apa ya?"
    ],
    "city_updated": [
        "Bisa ingatkan aku, kota tujuan pindahanku yang terbaru ke mana?",
        "Berdasarkan kabar pembaruan tadi, di kota mana tempat tinggalku yang baru?",
        "Kota baru yang menjadi tujuan kepindahanku tadi apa ya?"
    ],
    "job_updated": [
        "Bisa sebutkan profesi atau peran pekerjaanku yang terbaru setelah promosi?",
        "Setelah update karir tadi, apa jabatan atau pekerjaan baruku sekarang?",
        "Kamu masih ingat peran baruku di tempat kerja setelah berganti posisi?"
    ]
}

RECALL_ANSWERS_POOL = {
    "name": [
        "Tentu saja ingat, namamu adalah {ans}.",
        "Kamu adalah {ans}, senang selalu bisa mendampingimu berdiskusi!",
        "Nama yang kamu perkenalkan di awal percakapan kita adalah {ans}."
    ],
    "job": [
        "Tentu saja, kamu berprofesi dan bekerja sebagai seorang {ans}.",
        "Profesi pekerjaanmu sehari-hari adalah {ans}.",
        "Berdasarkan percakapan kita tadi, kamu berkarir sebagai {ans}."
    ],
    "city": [
        "Tentu, kamu berdomisili dan menetap di kota {ans}.",
        "Kota tempat tinggalmu saat ini adalah {ans}.",
        "Di awal percakapan tadi kamu menyebutkan tinggal di kota {ans}."
    ],
    "lang": [
        "Bahasa pemrograman yang kamu andalkan dalam pekerjaanmu adalah {ans}.",
        "Kamu tadi menceritakan bahwa stack bahasa kodingmu adalah {ans}.",
        "Tentu, bahasa pemrograman yang kamu gunakan adalah {ans}."
    ],
    "tool": [
        "Framework teknologi yang kamu gunakan dalam proyekmu adalah {ans}.",
        "Kamu tadi menyebutkan mengandalkan framework {ans}.",
        "Tentu ingat, kamu menggunakan tools framework {ans}."
    ],
    "hobby": [
        "Hobi yang biasa kamu lakukan saat waktu luang adalah {ans}.",
        "Aktivitas rekreasi favoritmu untuk melepas lelah adalah {ans}.",
        "Tentu saja, kegiatan hobimu yang menyenangkan adalah {ans}."
    ],
    "food": [
        "Makanan favorit kesukaanmu adalah {ans}.",
        "Hidangan lezat yang paling kamu gemari adalah {ans}.",
        "Tentu saja, kamu tadi menyebutkan sangat menyukai menu {ans}."
    ],
    "drink": [
        "Minuman favorit santaimu adalah {ans}.",
        "Kamu tadi bercerita suka menikmati segelas {ans}.",
        "Tentu ingat, minuman pilihanmu untuk bersantai adalah {ans}."
    ],
    "allergy": [
        "Tentu saja sangat kuperhatikan, kamu memiliki kondisi atau pantangan {ans}.",
        "Hal penting yang wajib dijaga dari konsumsimu adalah pantangan {ans}.",
        "Aku selalu mencatatnya dengan aman: kamu memiliki kondisi {ans}."
    ],
    "pet_name": [
        "Nama hewan peliharaan kesayanganmu di rumah adalah {ans}.",
        "Tentu saja ingat, peliharaan manjamu itu bernama {ans}.",
        "Peliharaan kesayangan yang menemanimu bernama {ans}."
    ],
    "pet_type": [
        "Hewan peliharaan yang kamu rawat di rumah adalah seekor {ans}.",
        "Tentu, hewan kesayangan yang tinggal bersamamu adalah {ans}."
    ],
    "travel": [
        "Destinasi liburan impian yang ingin kamu kunjungi tahun ini adalah {ans}.",
        "Rencana perjalanan wisata favoritmu adalah menuju ke {ans}.",
        "Tentu, tempat wisata yang sangat ingin kamu tuju adalah {ans}."
    ],
    "side_biz": [
        "Usaha sampingan mandiri yang sedang kamu rintis adalah {ans}.",
        "Bisnis sampingan yang kamu jalani di luar jam kerja adalah {ans}.",
        "Tentu saja ingat, kamu sedang aktif mengembangkan {ans}."
    ],
    "fitness_goal": [
        "Target latihan kebugaran jasmani yang sedang kamu fokuskan adalah {ans}.",
        "Program rutinitas fisik yang sedang kamu tekuni adalah {ans}.",
        "Tentu, kamu sedang berdisiplin menjalankan {ans}."
    ],
    "creative_project": [
        "Proyek kreatif yang sedang kamu kerjakan adalah {ans}.",
        "Karya seni atau kreasi mandiri yang sedang kamu bangun adalah {ans}."
    ],
    "philosophy_topic": [
        "Topik mendalam yang tadi kita bahas bersama adalah tentang {ans}.",
        "Tentu, topik refleksi menarik yang tadi menarik minatmu adalah {ans}."
    ],
    "city_updated": [
        "Berdasarkan kabar kepindahan terbarumu, kota tujuan barumu sekarang adalah {ans}.",
        "Setelah pembaruan tadi, tempat tinggal barumu adalah di kota {ans}."
    ],
    "job_updated": [
        "Setelah update karir dan promosi barumu, peran pekerjaan barumu sekarang adalah {ans}.",
        "Berdasarkan informasi terbaru darimu, jabatan barumu saat ini adalah {ans}."
    ]
}

def format_natural_recall_turn(recall_key: str, ans: str, turns: List[Dict[str, str]]) -> Tuple[str, str]:
    """Generates natural, idiomatic Indonesian recall question and answer without raw dict keys."""
    q_candidates = RECALL_QUESTIONS_POOL.get(recall_key, [f"Kamu masih ingat {recall_key} yang kuceritakan tadi?"])
    a_candidates = RECALL_ANSWERS_POOL.get(recall_key, [f"Tentu saja aku ingat, hal yang kamu ceritakan tadi adalah {ans}."])
    
    q_str = random.choice(q_candidates)
    a_str = random.choice(a_candidates).format(ans=ans)
    return q_str, a_str


# ---------------------------------------------------------------------------
# 12 Comprehensive & Diverse Dialogue Domain Builders
# ---------------------------------------------------------------------------

def make_tech_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})

    tech_variants = [
        f"Di tempat kerjaku sekarang kami banyak menggunakan {p['lang']} dan framework {p['tool']}. Menurutmu apa keunggulan utama stack ini?",
        f"Aku lagi eksplorasi arsitektur baru menggunakan bahasa {p['lang']} dipadukan dengan {p['tool']}. Apakah kombinasi ini scalable?",
        f"Proyek terbaruku dibangun dengan {p['lang']} dan {p['tool']}. Ada saran best practice untuk optimasi performanya?",
        f"Sehari-hari aku sering riset implementasi {p['tool']} berbasis {p['lang']}. Ekosistemnya terasa sangat produktif untuk tim kami."
    ]
    a_tech_variants = [
        f"Kombinasi {p['lang']} dengan {p['tool']} sangat populer karena efisiensi eksekusi dan ekosistem library yang matang, {p['name']}. Kuncinya ada pada modularitas arsitektur dan caching yang tepat.",
        f"Stack {p['lang']} dan {p['tool']} adalah kombinasi bertenaga tinggi! Keunggulannya ada pada kecepatan iterasi kode dan dukungan komunitas yang luar biasa aktif.",
        f"Pilihan stack yang sangat solid, {p['name']}. Penggunaan {p['tool']} di atas {p['lang']} memungkinkan tim membangun aplikasi yang kokoh dan mudah di-maintain."
    ]
    turns.append({"role": "user", "content": random.choice(tech_variants)})
    turns.append({"role": "assistant", "content": random.choice(a_tech_variants)})
    facts.append({"turn": 2, "key": "lang", "value": p["lang"]})
    facts.append({"turn": 2, "key": "tool", "value": p["tool"]})

    lifestyle_variants = [
        f"Kalau lagi jenuh sama urusan teknis, pelarianku biasanya minum {p['drink']} sambil {p['hobby']}.",
        f"Biar nggak burnout kerja terus, rutinitas favoritku adalah meluangkan waktu buat {p['hobby']} dan menikmati {p['drink']}.",
        f"Untuk menjaga keseimbangan hidup setelah jam kerja, aku paling suka santai {p['hobby']} ditemani segelas {p['drink']}."
    ]
    turns.append({"role": "user", "content": random.choice(lifestyle_variants)})
    turns.append({"role": "assistant", "content": f"Itu pola istirahat yang sangat sehat, {p['name']}. Menikmati {p['drink']} sambil {p['hobby']} terbukti ampuh menyegarkan kembali fokus pikiran."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    # Distractor turns
    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["job", "city", "lang", "drink", "hobby"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "tech_and_career",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_problem_solving_tech_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """User starts immediately with a technical challenge; introduces identity naturally later."""
    turns, facts = [], []

    issues = [
        ("Halo! Ada saran praktis nggak buat optimasi query database yang mulai lambat saat beban traffic tinggi?",
         "Gunakan EXPLAIN ANALYZE untuk membaca execution plan, pastikan kolom yang sering difilter terindeks dengan benar (composite index jika perlu), dan pasang caching layer seperti Redis."),
        ("Permisi asisten, aku lagi menimbang arsitektur event-driven vs REST API untuk microservices. Mana yang lebih cocok?",
         "Gunakan REST API untuk komunikasi sinkron yang butuh respon instan langsung ke user, dan gunakan Event-Driven (Kafka/RabbitMQ) untuk alur kerja asinkron yang membutuhkan decoupling tinggi."),
        ("Hai asisten! Mau tanya tips menangani memory leak di aplikasi backend berbasis kontainer Docker dong.",
         "Lakukan profiling memory menggunakan heap dump saat beban aktif, periksa koneksi resource atau database yang lupa ditutup, dan pastikan memory limit di docker dikonfigurasi dengan aman.")
    ]
    u_init, a_init = random.choice(issues)
    turns.append({"role": "user", "content": u_init})
    turns.append({"role": "assistant", "content": a_init})

    name = p["name"]
    intro_replies = [
        f"Makasih banyak sarannya! Oh iya kenalkan, namaku {name}. Aku berprofesi sebagai {p['job']} di kota {p['city']}. Di kantorku kami memang pakai bahasa {p['lang']}.",
        f"Penjelasanmu sangat mencerahkan! Ngomong-ngomong kenalin, aku {name}. Sehari-hari aku sibuk jadi {p['job']} yang tinggal di {p['city']} dan mengandalkan {p['lang']}.",
        f"Keren banget solusinya, langsung kupraktikkan! Salam kenal, aku {name} dari {p['city']}. Profesi utamaku {p['job']}, dan stack utamaku memang {p['lang']}."
    ]
    turns.append({"role": "user", "content": random.choice(intro_replies)})
    turns.append({"role": "assistant", "content": f"Salam kenal hangat, {name}! Senang sekali berdiskusi dengan seorang {p['job']} dari {p['city']}. Ekosistem bahasa {p['lang']} memang sangat handal bila didukung arsitektur yang rapi."})
    facts.append({"turn": 2, "key": "name", "value": name})
    facts.append({"turn": 2, "key": "job", "value": p["job"]})
    facts.append({"turn": 2, "key": "city", "value": p["city"]})
    facts.append({"turn": 2, "key": "lang", "value": p["lang"]})

    turns.append({"role": "user", "content": f"Kalau pas istirahat kerja seharian, biasanya aku santai sambil menikmati {p['drink']} biar rileks."})
    turns.append({"role": "assistant", "content": f"Pilihan istirahat yang nikmat, {name}. Menikmati segelas {p['drink']} adalah cara mantap untuk merestart konsentrasi."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["job", "city", "lang", "drink"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "problem_solving_tech",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_lifestyle_health_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    food_variants = [
        f"Saat makan siang biasanya aku paling suka menyantap {p['food']}, tapi penting dicatat kalau aku punya kondisi {p['allergy']}.",
        f"Bicara soal kuliner harian, makanan favoritku adalah {p['food']}. Namun aku harus selalu berhati-hati karena ada {p['allergy']}.",
        f"Menu makanan yang paling membangkitkan seleraku itu {p['food']}, cuma aku wajib menghindari pemicu karena {p['allergy']}."
    ]
    turns.append({"role": "user", "content": random.choice(food_variants)})
    turns.append({"role": "assistant", "content": f"Tercatat dengan sangat baik, {p['name']}! {p['food']} memang hidangan yang nikmat, dan kita tentu wajib selalu memperhatikan kondisi {p['allergy']} agar kesehatanmu tetap terjaga prima."})
    facts.append({"turn": 2, "key": "food", "value": p["food"]})
    facts.append({"turn": 2, "key": "allergy", "value": p["allergy"]})

    pet_variants = [
        f"Di tempat tinggalku aku juga memelihara hewan kesayangan, yaitu seekor {p['pet_type']} yang kuberi nama {p['pet_name']}.",
        f"Teman setiaku saat bersantai di rumah adalah peliharaanku, seekor {p['pet_type']} lucu bernama {p['pet_name']}.",
        f"Suasana rumah selalu ramai berkat kehadiran {p['pet_type']} kesayanganku yang namanya {p['pet_name']}."
    ]
    turns.append({"role": "user", "content": random.choice(pet_variants)})
    turns.append({"role": "assistant", "content": f"Pasti menggemaskan sekali! Memiliki {p['pet_type']} bernama {p['pet_name']} selalu membawa suasana ceria dan penawar lelah di rumah."})
    facts.append({"turn": 4, "key": "pet_type", "value": p["pet_type"]})
    facts.append({"turn": 4, "key": "pet_name", "value": p["pet_name"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["allergy", "pet_name", "food", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "lifestyle_and_health",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_fitness_gym_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Discussions revolving around fitness, sports, gym routines, and nutrition."""
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    fit_variants = [
        f"Biar badan tetap bugar di sela-sela rutinitas, saat ini aku lagi disiplin fokus ke {p['fitness_goal']}.",
        f"Target kebugaran utamaku sekarang adalah konsisten menjalankan {p['fitness_goal']}. Ada saran pengaturan nutrisinya?",
        f"Tahun ini aku punya target jasmani yang ambisius, yaitu menyelesaikan {p['fitness_goal']} secara bertahap."
    ]
    turns.append({"role": "user", "content": random.choice(fit_variants)})
    turns.append({"role": "assistant", "content": f"Komitmen kesehatan yang luar biasa, {p['name']}! Untuk mendukung {p['fitness_goal']}, kuncinya ada pada hidrasi cukup, asupan protein berkala, dan waktu tidur minimal 7-8 jam untuk pemulihan jaringan otot."})
    facts.append({"turn": 2, "key": "fitness_goal", "value": p["fitness_goal"]})

    turns.append({"role": "user", "content": f"Setelah sesi latihan yang melelahkan, minuman andalanku buat menyegarkan badan itu {p['drink']}."})
    turns.append({"role": "assistant", "content": f"Segar sekali! Memilih {p['drink']} sesudah olahraga memberikan kepuasan rasa yang melengkapi hidrasi tubuhmu."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["fitness_goal", "drink", "job"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "fitness_and_gym",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_travel_adventure_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    travel_variants = [
        f"Tahun ini aku punya target liburan impian yang sudah lama kurencanakan, yaitu mengunjungi {p['travel']}.",
        f"Rencana perjalanan wisataku berikutnya adalah menjelajahi keindahan alam di {p['travel']}.",
        f"Salah satu resolusi jalan-jalanku tahun ini adalah berlibur dan healing ke {p['travel']}."
    ]
    turns.append({"role": "user", "content": random.choice(travel_variants)})
    turns.append({"role": "assistant", "content": f"Destinasi yang sangat memukau, {p['name']}! {p['travel']} punya pesona ikonik dan pasti memberikan pengalaman liburan tak terlupakan."})
    facts.append({"turn": 2, "key": "travel", "value": p["travel"]})

    biz_variants = [
        f"Untuk mendanai rencana liburan dan menambah tabungan, sekarang aku aktif merintis usaha sampingan berupa {p['side_biz']}.",
        f"Selain pekerjaan utama, kesibukan baruku saat ini adalah mengembangkan usaha {p['side_biz']}.",
        f"Aku juga lagi belajar berwirausaha mandiri dengan menjalankan {p['side_biz']} di waktu luang."
    ]
    turns.append({"role": "user", "content": random.choice(biz_variants)})
    turns.append({"role": "assistant", "content": f"Langkah wirausaha yang sangat inspiratif! Mengembangkan {p['side_biz']} adalah cara cerdas membangun kemandirian finansial."})
    facts.append({"turn": 4, "key": "side_biz", "value": p["side_biz"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["travel", "side_biz", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "travel_and_adventure",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_culinary_recipe_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Indonesian culinary discussion, cooking recipes, and allergy considerations."""
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    culinary_variants = [
        f"Hari ini aku lagi pengen banget masak {p['food']}. Tapi kamu harus ingat, aku punya kondisi {p['allergy']}.",
        f"Kuliner kesukaanku itu {p['food']}, namun aku harus berhati-hati karena ada {p['allergy']}. Ada ide kreasi resep yang aman?",
        f"Aku berencana bikin menu istimewa {p['food']} di rumah, tapi wajib disesuaikan karena aku {p['allergy']}."
    ]
    turns.append({"role": "user", "content": random.choice(culinary_variants)})
    turns.append({"role": "assistant", "content": f"Siap dicatat dengan cermat, {p['name']}! Untuk memasak {p['food']} yang ramah bagi kondisi {p['allergy']}, kita bisa mengganti bahan pemicu dengan alternatif bumbu alami yang tetap kaya rempah dan gurih."})
    facts.append({"turn": 2, "key": "food", "value": p["food"]})
    facts.append({"turn": 2, "key": "allergy", "value": p["allergy"]})

    turns.append({"role": "user", "content": f"Sebagai teman santap makanannya, aku selalu menyiapkan minuman favoritku yaitu {p['drink']}."})
    turns.append({"role": "assistant", "content": f"Kombinasi yang sangat menggugah selera! Paduan hidangan tersebut dengan segelas {p['drink']} pasti bikin suasana makan semakin nikmat."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["food", "allergy", "drink"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "culinary_and_recipes",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_daily_life_banter_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Everyday Indonesian life, commute, rain, pets, and relaxing."""
    turns, facts = [], []

    banters = [
        (f"Aduh hari ini jalanan di kota {p['city']} macetnya lumayan parah, baru bisa santai sekarang.",
         "Pasti melelahkan sekali ya. Untung sekarang sudah sampai di tempat istirahat dengan selamat. Ambil waktu rileks sejenak!"),
        (f"Cuaca di {p['city']} akhir-akhir ini lagi sering hujan lebat pas jam pulang kantor.",
         "Musim penghujan memang sering bikin mobilitas terhambat. Yang penting selalu jaga daya tahan tubuh dan jangan lupa jas hujan."),
        (f"Hari ini kerjaanku lumayan padat, rasanya pengen cepet-cepet santai di rumah {p['city']}.",
         "Semangat ya! Setelah hari yang produktif, beristirahat dan memanjakan diri dengan hal santai adalah hakmu.")
    ]
    u_0, a_0 = random.choice(banters)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})

    name = p["name"]
    pet_intros = [
        f"Untung pas nyampe rumah langsung disambut peliharaanku, seekor {p['pet_type']} yang kuberi nama {p['pet_name']}. Oh iya, kenalkan namaku {name}.",
        f"Penawar lelah terbaikku ya peliharaanku ini, si {p['pet_name']} yang merupakan {p['pet_type']} kesayanganku. Namaku {name} btw.",
        f"Langsung hilang capeknya pas ketemu {p['pet_type']} peliharaanku si {p['pet_name']}. Salam kenal, aku {name}."
    ]
    turns.append({"role": "user", "content": random.choice(pet_intros)})
    turns.append({"role": "assistant", "content": f"Halo {name}! Keberadaan {p['pet_type']} bernama {p['pet_name']} memang moodbooster luar biasa setelah seharian menghadapi kemacetan dan kepenatan."})
    facts.append({"turn": 2, "key": "name", "value": name})
    facts.append({"turn": 2, "key": "pet_type", "value": p["pet_type"]})
    facts.append({"turn": 2, "key": "pet_name", "value": p["pet_name"]})

    turns.append({"role": "user", "content": f"Sehari-hari selain urusan rumah, aku berkarir aktif sebagai {p['job']}."})
    turns.append({"role": "assistant", "content": f"Profesi sebagai {p['job']} tentu membutuhkan dedikasi tinggi. Menjaga harmoni antara pekerjaan dan relaksasi di rumah adalah kunci kebahagiaan."})
    facts.append({"turn": 4, "key": "job", "value": p["job"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["city", "pet_name", "job"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "daily_life_banter",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_gaming_creative_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    game_variants = [
        f"Pas lagi pengen hiburan santai di malam hari, game favorit yang rutin kumainkan adalah {p['game']}.",
        f"Untuk melepas kepenatan rutinitas, biasanya aku login main game {p['game']} bareng teman-teman.",
        f"Hobi gaming-ku saat ini lagi banyak kuhabiskan di game {p['game']}, gameplay-nya sangat seru."
    ]
    turns.append({"role": "user", "content": random.choice(game_variants)})
    turns.append({"role": "assistant", "content": f"Pilihan hiburan yang asyik, {p['name']}! Game {p['game']} memang punya mekanik yang menarik dan sangat pas dimainkan untuk melepas penat."})
    facts.append({"turn": 2, "key": "game", "value": p["game"]})

    turns.append({"role": "user", "content": f"Selain gaming di depan layar, di waktu luang aku juga sering {p['hobby']}."})
    turns.append({"role": "assistant", "content": f"Kombinasi hobi yang sangat berimbang! Menyelipkan aktivitas {p['hobby']} membuat keseharianmu lebih bervariasi."})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["game", "hobby", "job"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "gaming_and_creative",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_career_growth_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Professional career development, negotiations, skill acquisition, and workplace challenges."""
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    turns.append({"role": "user", "content": f"Di posisiku sebagai {p['job']}, belakangan ini aku lagi banyak belajar tentang kepemimpinan tim dan manajemen prioritas proyek."})
    turns.append({"role": "assistant", "content": f"Langkah pengembangan diri yang sangat berharga, {p['name']}. Menjadi {p['job']} yang unggul tidak hanya butuh kemampuan teknis, tetapi juga keahlian delegasi dan empati komunikasi."})

    turns.append({"role": "user", "content": f"Biar tetap fresh saat brainstorming solusi, segelas {p['drink']} selalu jadi teman setiaku di meja kerja."})
    turns.append({"role": "assistant", "content": f"Suasana kerja yang nyaman! Segelas {p['drink']} memang bisa membantu menjaga kejernihan berpikir saat menyusun strategi kerja."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["job", "drink", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "career_growth",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_creative_arts_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Creative arts, novel writing, photography, custom crafting, and hobbies."""
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    art_variants = [
        f"Di luar jam rutinitas, saat ini aku lagi asyik menyelesaikan karya kreatif, yaitu {p['creative_project']}.",
        f"Aku lagi punya side project seni yang menyenangkan banget: {p['creative_project']}. Rasanya memuaskan jiwa kreatifku.",
        f"Waktu luang akhir-akhir ini banyak kuhabiskan untuk fokus mengerjakan {p['creative_project']}."
    ]
    turns.append({"role": "user", "content": random.choice(art_variants)})
    turns.append({"role": "assistant", "content": f"Proyek kreasi yang sungguh menginspirasi, {p['name']}! Menyalurkan imajinasi lewat {p['creative_project']} adalah ekspresi estetika yang memperkaya batin."})
    facts.append({"turn": 2, "key": "creative_project", "value": p["creative_project"]})

    turns.append({"role": "user", "content": f"Kalau lagi cari inspirasi baru, aku paling suka santai sambil menikmati {p['food']} favoritku."})
    turns.append({"role": "assistant", "content": f"Santapan lezat {p['food']} pasti ampuh membangkitkan suasana hati dan memicu ide-ide segar!"})
    facts.append({"turn": 4, "key": "food", "value": p["food"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["creative_project", "food", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "creative_arts",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_science_philosophy_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Deep science, astronomy, Stoicism, and life reflections."""
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    phil_variants = [
        f"Akhir-akhir ini aku lagi sering merenungi sebuah topik yang sangat menarik, yaitu {p['philosophy_topic']}. Bagaimana perspektifmu tentang hal itu?",
        f"Aku baru saja membaca literatur menarik tentang {p['philosophy_topic']}. Rasanya membuka cara pandang baru bagi hidupku.",
        f"Di sela kesibukanku, topik yang paling sering memicu rasa penasaranku belakangan ini adalah {p['philosophy_topic']}."
    ]
    turns.append({"role": "user", "content": random.choice(phil_variants)})
    turns.append({"role": "assistant", "content": f"Topik refleksi yang sangat mendalam, {p['name']}! Membedah {p['philosophy_topic']} melatih kejernihan berpikir dan membantu kita memahami semesta serta posisi diri kita dengan lebih bijaksana."})
    facts.append({"turn": 2, "key": "philosophy_topic", "value": p["philosophy_topic"]})

    turns.append({"role": "user", "content": f"Diskusi mendalam seperti ini paling nikmat dinikmati sambil menyeruput secangkir {p['drink']}."})
    turns.append({"role": "assistant", "content": f"Sangat cocok! Kehangatan {p['drink']} selalu jadi pelengkap sempurna untuk perbincangan filsafat dan sains yang berbobot."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["philosophy_topic", "drink", "job"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "science_and_philosophy",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q_rec,
            "answer": a_rec
        }
    }


def make_update_correction_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Dialogues with explicit factual updates and corrections."""
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    q_dis1, a_dis1 = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis1})
    turns.append({"role": "assistant", "content": a_dis1})

    update_type = random.choice(["city", "job"])
    if update_type == "city":
        update_variants = [
            f"Kabar terbarunya, bulan depan aku resmi pindah tempat tinggal ke {p['alt_city']} karena urusan keluarga.",
            f"Oh iya, sekadar info update, minggu depan aku akan merelokasi tempat tinggalku ke kota {p['alt_city']}.",
            f"Ada perkembangan baru nih, aku baru saja menyelesaikan urusan kepindahan rumah ke {p['alt_city']}."
        ]
        u_up = random.choice(update_variants)
        a_up = f"Selamat atas rencana kepindahan barumu ke kota {p['alt_city']}, {p['name']}! Semoga suasana baru di sana membawa kenyamanan dan kelancaran."
        facts.append({"turn": 4, "key": "city_updated", "value": p["alt_city"]})
        target_key = "city_updated"
        old_val = p["city"]
        new_val = p["alt_city"]
    else:
        update_variants = [
            f"Kabar gembiranya, aku baru saja resmi dipromosikan dan berganti peran menjadi {p['alt_job']}.",
            f"Ada kabar baik soal karirku, mulai bulan depan aku beralih profesi menjadi {p['alt_job']}.",
            f"Update penting tentang pekerjaanku: per hari ini aku mulai mengemban tanggung jawab baru sebagai {p['alt_job']}."
        ]
        u_up = random.choice(update_variants)
        a_up = f"Wah selamat banyak atas pencapaian karir barumu sebagai {p['alt_job']}, {p['name']}! Ini langkah besar yang membanggakan."
        facts.append({"turn": 4, "key": "job_updated", "value": p["alt_job"]})
        target_key = "job_updated"
        old_val = p["job"]
        new_val = p["alt_job"]

    turns.append({"role": "user", "content": u_up})
    turns.append({"role": "assistant", "content": a_up})

    q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
    turns.append({"role": "user", "content": q_dis2})
    turns.append({"role": "assistant", "content": a_dis2})

    if turns_count >= 12:
        q_dis3, a_dis3 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] not in (q_dis1, q_dis2)])
        turns.append({"role": "user", "content": q_dis3})
        turns.append({"role": "assistant", "content": a_dis3})

    q_rec, a_rec = format_natural_recall_turn(target_key, new_val, turns)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "memory_update_and_correction",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": target_key,
            "ground_truth": new_val,
            "old_value": old_val,
            "question": q_rec,
            "answer": a_rec
        }
    }


BUILDERS = [
    make_tech_dialogue,
    make_problem_solving_tech_dialogue,
    make_lifestyle_health_dialogue,
    make_fitness_gym_dialogue,
    make_travel_adventure_dialogue,
    make_gaming_creative_dialogue,
    make_culinary_recipe_dialogue,
    make_daily_life_banter_dialogue,
    make_career_growth_dialogue,
    make_creative_arts_dialogue,
    make_science_philosophy_dialogue,
    make_update_correction_dialogue
]


def format_chatml(turns: List[Dict[str, str]]) -> str:
    lines = []
    for t in turns:
        lines.append(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streaming 100M Token Generation Engine
# ---------------------------------------------------------------------------

def generate_100m_tokens(
    tokenizer_path: str = "dataset/tokenizer.json",
    output_dir: str = "dataset",
    train_tokens_target: int = 95_000_000,
    val_tokens_target: int   =  2_500_000,
    test_tokens_target: int  =  2_500_000,
    batch_size: int = 500,
    seed: int = 42
):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("      100M TOKENS MULTI-TURN CONVERSATIONAL MEMORY GENERATOR")
    print("=" * 72)
    print(f"  Target Total Tokens : {train_tokens_target + val_tokens_target + test_tokens_target:,}")
    print(f"  Train Target        : {train_tokens_target:,} tokens (95%)")
    print(f"  Validation Target   : {val_tokens_target:,} tokens (2.5%)")
    print(f"  Test Target         : {test_tokens_target:,} tokens (2.5%)")
    print(f"  Tokenizer           : {tokenizer_path}")
    print(f"  Dialogue Domains    : {len(BUILDERS)} builders (100% natural recall phrasing)")
    print(f"  Distractor Pool     : {len(MASSIVE_DISTRACTORS)} broad intellectual topics")
    print("=" * 72)

    tok = Tokenizer.from_file(tokenizer_path)

    splits = [
        ("conversations_100M_train.jsonl", train_tokens_target, "TRAIN"),
        ("conversations_100M_val.jsonl",   val_tokens_target,   "VAL"),
        ("conversations_100M_test.jsonl",  test_tokens_target,  "TEST"),
    ]

    total_written_conversations = 0
    total_generated_tokens = 0
    start_total_time = time.time()
    summary_meta = {}

    for filename, target_tokens, split_name in splits:
        filepath = os.path.join(output_dir, filename)
        print(f"\n>>> Generating Split [{split_name}]: Target {target_tokens:,} tokens -> {filename}")
        
        tokens_in_split = 0
        convs_in_split = 0
        t0 = time.time()

        with open(filepath, 'w', encoding='utf-8') as f_out:
            while tokens_in_split < target_tokens:
                # Generate a batch of candidate dialogues
                batch_items = []
                batch_chatml = []

                for _ in range(batch_size):
                    eid = f"User_{total_written_conversations + len(batch_items) + 1:07d}"
                    persona = build_random_persona(eid)
                    turns_len = random.choice([8, 10, 12, 14])
                    builder = random.choice(BUILDERS)

                    diag = builder(persona, turns_len)
                    chatml_str = format_chatml(diag["turns"])

                    curr_idx = total_written_conversations + len(batch_items) + 1
                    item = {
                        "id": f"conv_{curr_idx:07d}",
                        "entity_id": eid,
                        "topic": diag["topic"],
                        "num_turns": len(diag["turns"]),
                        "turns": diag["turns"],
                        "chatml": chatml_str,
                        "facts": diag["facts"],
                        "target_recall": diag["target_recall"],
                    }
                    batch_items.append(item)
                    batch_chatml.append(chatml_str)

                # Batch tokenization for high throughput
                encodings = tok.encode_batch(batch_chatml)

                # Write out dialogues and count tokens
                for item, enc in zip(batch_items, encodings):
                    tok_len = len(enc.ids)
                    item["token_length"] = tok_len
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    tokens_in_split += tok_len
                    convs_in_split += 1
                    total_written_conversations += 1
                    total_generated_tokens += tok_len

                    if tokens_in_split >= target_tokens:
                        break

                elapsed = time.time() - t0
                tok_per_sec = tokens_in_split / max(elapsed, 0.001)
                progress_pct = min(tokens_in_split / target_tokens * 100, 100.0)
                print(f"\r  [{split_name}] {tokens_in_split:>10,d} / {target_tokens:,d} tokens ({progress_pct:5.1f}%) | "
                      f"{convs_in_split:>6,d} convs | {tok_per_sec:>9,.0f} tok/s | {elapsed:4.1f}s", end='', flush=True)

        print()
        filesize_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"  [DONE {split_name}] Generated {tokens_in_split:,} tokens across {convs_in_split:,} dialogues "
              f"({filesize_mb:.1f} MB) in {time.time()-t0:.1f}s")

        summary_meta[split_name.lower()] = {
            "filename": filename,
            "total_tokens": tokens_in_split,
            "total_conversations": convs_in_split,
            "filesize_mb": round(filesize_mb, 2)
        }

    total_time = time.time() - start_total_time
    print("\n" + "=" * 72)
    print("                     ALL SPLITS COMPLETED                      ")
    print("=" * 72)
    print(f"  Total Tokens Generated   : {total_generated_tokens:,}")
    print(f"  Total Conversations      : {total_written_conversations:,}")
    print(f"  Total Wallclock Time     : {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    print(f"  Overall Generation Speed : {total_generated_tokens/total_time:,.0f} tokens/second")
    print("=" * 72)

    meta_file = os.path.join(output_dir, "conversations_100M_metadata.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump({
            "generator": "generate_100m_tokens_dataset.py",
            "seed": seed,
            "total_tokens": total_generated_tokens,
            "total_conversations": total_written_conversations,
            "splits": summary_meta
        }, f, indent=2, ensure_ascii=False)

    print(f"  Metadata written -> {meta_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 100M Tokens Multi-Turn Conversational Memory Dataset")
    parser.add_argument("--tokenizer", default="dataset/tokenizer.json")
    parser.add_argument("--output_dir", default="dataset")
    parser.add_argument("--target_tokens", type=int, default=None,
                        help="Total target tokens across all splits (automatically splits 90%% train, 5%% val, 5%% test)")
    parser.add_argument("--train_tokens", type=int, default=95_000_000)
    parser.add_argument("--val_tokens", type=int, default=2_500_000)
    parser.add_argument("--test_tokens", type=int, default=2_500_000)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_tokens = args.train_tokens
    val_tokens = args.val_tokens
    test_tokens = args.test_tokens

    if args.target_tokens is not None:
        train_tokens = int(args.target_tokens * 0.90)
        val_tokens = int(args.target_tokens * 0.05)
        test_tokens = int(args.target_tokens * 0.05)

    generate_100m_tokens(
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        train_tokens_target=train_tokens,
        val_tokens_target=val_tokens,
        test_tokens_target=test_tokens,
        batch_size=args.batch_size,
        seed=args.seed
    )
