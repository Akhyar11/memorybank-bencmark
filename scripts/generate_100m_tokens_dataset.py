"""
generate_100m_tokens_dataset.py – Infinite-Diversity Multi-Turn Conversational Memory Dataset Generator.

Generates 100,000,000 tokens of rich, organic, multi-turn (4-16 turns) Indonesian dialogues:
- Train split: 95,000,000 tokens (~120,000 conversations)
- Val split:    2,500,000 tokens (~3,100 conversations)
- Test split:   2,500,000 tokens (~3,100 conversations)

Core Diversity Engine:
1. 15 Distinct Dialogue Archetypes:
   - Technical Architecture & Code Consulting (with real code snippets)
   - Bug Debugging & Traceback Diagnostics (with stack traces & error messages)
   - Indonesian Culinary Arts, Recipes & Allergen Safety
   - Travel Itinerary Planning & Regional Exploration
   - Fitness Periodization, Strength Training & Nutrition
   - Creative Writing, Sci-Fi Worldbuilding & Story Polish
   - Career Navigation, Promotion, 1-on-1s & Salary Negotiation
   - Deep Astronomy, Quantum Science & Physics
   - Stoic Philosophy, Mindset & Life Dilemmas
   - Everyday Banter, Commuting, Rainy Days & Cozy Cafes
   - Pet Care, Behavior & Funny Animal Anecdotes
   - Personal Finance, Emergency Funds & Wealth Building
   - Gaming Meta, Esports & Indie Game Discussions
   - Book & Cinema Story Analysis
   - Dynamic Memory Update & Fact Correction
2. Combinatorial Clause Synthesizer (Stochastic Grammar):
   Every turn is dynamically constructed from independent syntactic modules
   (Discourse Opener + Contextual Stance + Topic Clause + Pragmatic Inquiry),
   generating billions of non-repeating, natural Indonesian sentences.
3. Multi-Format Assistant Responses:
   Includes bulleted lists, numbered step-by-step guides, code blocks (Python, SQL, Go, JS, Bash),
   pros/cons comparisons, and warm conversational paragraphs.
4. Natural Episodic Memory Planting & Recall:
   Facts are dropped organically across turns (intro, constraint, incidental detail, or correction),
   and tested via direct, contextual, and cross-attribute questions with 100% natural Indonesian phrasing.
"""
import os
import sys
import json
import time
import random
import argparse
from typing import List, Dict, Any, Tuple
from tokenizers import Tokenizer


# ===========================================================================
# 1. EXPANDED ENTITIES & KNOWLEDGE BASES
# ===========================================================================

NAMES = [
    # Male & Female names from all Indonesian regions (Java, Sunda, Batak, Minang, Bali, Eastern, Modern)
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
    "Permata", "Qori", "Rahma", "Safira", "Tari", "Ulfa", "Wulan", "Yasmine", "Zelva", "Amira",
    "Bintang", "Cahaya", "Damar", "Emir", "Fadhil", "Gibran", "Harun", "Iskandar", "Jonathan", "Kusno",
    "Latief", "Mansur", "Nadhif", "Osman", "Prakoso", "Qasim", "Rasyid", "Sabda", "Tunggul", "Vicky",
    "Wira", "Yafiq", "Zulfikar", "Almira", "Binar", "Clara", "Dara", "Evita", "Fatimah", "Ghea",
    "Hafsa", "Inara", "Jasmin", "Keisha", "Lavanya", "Marsya", "Najwa", "Odelia", "Prita", "Qonita",
    "Rania", "Salma", "Talitha", "Ufaira", "Vania", "Widya", "Xenia", "Yasmin", "Zaskia", "Aurelia",
    "Bonaventura", "Christo", "Dion", "Efraim", "Felix", "Geraldo", "Hutomo", "Ignatius", "Julian", "Krisna",
    "Lambertus", "Marcel", "Nathan", "Octavianus", "Patrick", "Reynold", "Stefanus", "Timothy", "Valerian", "Willy",
    "Tegar", "Sandi", "Zidan", "Daffa", "Rafi", "Kezia", "Sherly", "Tasya", "Nabila", "Syifa",
    "Alif", "Reyhan", "Hanif", "Dzaki", "Farras", "Arkan", "Haikal", "Kenzo", "Azka", "Atta"
]

CITIES = [
    # Java
    "Jakarta Selatan", "Jakarta Pusat", "Jakarta Barat", "Surabaya", "Bandung", "Semarang", "Yogyakarta", "Malang", "Solo", "Bogor", "Bekasi",
    "Tangerang", "Depok", "Cimahi", "Cirebon", "Sukabumi", "Tasikmalaya", "Pekalongan", "Tegal",
    "Magelang", "Purwokerto", "Kediri", "Blitar", "Madiun", "Jember", "Banyuwangi", "Salatiga",
    "Kudus", "Mojokerto", "Pasuruan", "Probolinggo", "Tuban", "Lamongan", "Cilacap", "Garut",
    # Sumatra
    "Medan", "Palembang", "Padang", "Banda Aceh", "Pekanbaru", "Dumai", "Jambi", "Bengkulu",
    "Bandar Lampung", "Pangkalpinang", "Batam", "Tanjungpinang", "Pematangsiantar", "Binjai",
    "Bukittinggi", "Lubuklinggau", "Prabumulih", "Metro",
    # Kalimantan
    "Banjarmasin", "Pontianak", "Balikpapan", "Samarinda", "Tarakan", "Palangkaraya", "Banjarbaru", "Singkawang",
    # Sulawesi
    "Makassar", "Manado", "Palu", "Kendari", "Gorontalo", "Ambon", "Ternate", "Bitung", "Parepare", "Palopo", "Baubau",
    # Bali, Nusa Tenggara, Maluku, Papua
    "Denpasar", "Singaraja", "Mataram", "Kupang", "Labuan Bajo", "Jayapura", "Sorong", "Manokwari", "Merauke", "Timika"
]

TECH_ROLES = [
    "Frontend Developer", "Backend Developer", "Fullstack Engineer", "Data Scientist",
    "Machine Learning Engineer", "DevOps Engineer", "Mobile Developer (iOS/Android)", "UI/UX Designer",
    "Product Manager", "QA Automation Engineer", "Cybersecurity Analyst", "Cloud Solutions Architect",
    "Database Administrator (DBA)", "Site Reliability Engineer (SRE)", "AI Prompt & Evaluation Engineer",
    "Blockchain & Smart Contract Developer", "Enterprise Systems Analyst", "Scrum Master & Agile Coach",
    "Embedded Systems & IoT Engineer", "Network Operations Specialist", "Data Engineer / ETL Specialist",
    "Security Operations Center (SOC) Analyst", "Platform & Infrastructure Engineer", "3D Game Developer",
    "Computer Vision Researcher", "NLP & LLM Systems Engineer", "Firmware Engineer", "Solutions Architect"
]

NON_TECH_ROLES = [
    "Dokter Umum", "Dokter Gigi Spesialis", "Apoteker Farmasi", "Arsitek Bangunan Tropis",
    "Guru Matematika SMA", "Dosen Ilmu Komunikasi", "Akuntan Publik Bersertifikat", "Konsultan Keuangan Mandiri",
    "Pengacara Hukum Bisnis", "Jurnalis Investigasi", "Fotografer Komersial & Produk", "Executive Chef Restoran",
    "Editor Video Kreatif", "Desainer Interior Hunian", "Brand Marketing Strategist", "Psikolog Klinis Dewasa",
    "Penerjemah Bahasa Tersumpah", "Copywriter & Penulis Konten", "Barista & Roaster Spesialis", "Fisioterapis Olahraga",
    "Perencana Tata Kota", "Spesialis Manajemen Logistik", "HR Talent Acquisition", "Kurator Galeri Seni Kontemporer",
    "Manajer Operasional Hotel", "Instruktur Kebugaran Pribadi", "Analis Kebijakan Publik", "Penyiar Radio & Podcaster"
]

PROGRAMMING_LANGS = [
    "Python", "TypeScript", "JavaScript", "Golang", "Rust", "Java", "Kotlin",
    "Swift", "C++", "PHP", "Dart", "C#", "SQL", "Scala", "Ruby", "Elixir"
]

FRAMEWORKS_AND_TOOLS = [
    "React", "Vue.js", "Next.js", "FastAPI", "Django", "Node.js (Express)", "Flutter",
    "PyTorch", "Docker", "Kubernetes", "Spring Boot", "Laravel", "NestJS", "TailwindCSS",
    "PostgreSQL", "Redis", "Apache Kafka", "MongoDB", "Elasticsearch", "Terraform", "GraphQL",
    "RabbitMQ", "Apache Spark", "GitLab CI/CD", "AWS Lambda", "Supabase", "Prisma ORM"
]

HOBBIES = [
    "bermain futsal bersama kawan lama", "bersepeda santai keliling kota di pagi hari",
    "jogging pagi di taman kota", "bermain game RPG open-world", "membaca novel fiksi ilmiah dan sejarah",
    "fotografi jalanan (street photography)", "bermain gitar akustik lagu-lagu santai",
    "belajar memasak aneka kue dan artisan sourdough", "merawat tanaman hias indoor monstera",
    "latihan angkat beban di gym", "berenang santai gaya bebas", "menonton film dokumenter sains dan alam",
    "mendaki gunung dan menyusuri jalur setapak", "camping di alam terbuka menikmati api unggun",
    "bermain catur kilat online", "menulis opini dan ulasan di blog pribadi", "belajar bahasa asing secara otodidak",
    "melukis cat air pemandangan alam", "merakit keyboard mekanikal custom", "bermain bulutangkis di akhir pekan",
    "merekam podcast obrolan santai", "merestorasi motor klasik antik", "bermain skateboard di skatepark"
]

GAMES = [
    "Valorant", "Mobile Legends: Bang Bang", "Genshin Impact", "Dota 2", "Minecraft survival mode",
    "EA Sports FC 24", "Elden Ring", "PUBG Mobile", "Apex Legends", "The Witcher 3: Wild Hunt",
    "Honkai: Star Rail", "Cyberpunk 2077", "Stardew Valley", "Free Fire MAX", "Zelda: Tears of the Kingdom",
    "Black Myth: Wukong", "Baldur's Gate 3", "Counter-Strike 2", "Palworld", "Hades II", "Monster Hunter World"
]

FOODS = [
    "Nasi Goreng spesial babat gongso", "Sate Ayam Madura bumbu kacang kental", "Rendang Sapi khas Minang gurih",
    "Mie Ayam Bakso urat jumbo", "Gado-gado siram saus kacang medok", "Soto Betawi kuah santan gurih",
    "Ayam Geprek sambal korek bawang", "Nasi Padang komplit lauk cincang", "Pempek Kapal Selam Palembang cuko pekat",
    "Rawon Daging sapi Surabaya hitam keluak", "Bakso Malang komplit tahu bakso", "Nasi Uduk Betawi semur tahu",
    "Bebek Sinjay Madura sambal mangga", "Ayam Betutu bumbu base genep Bali", "Gudeg Jogja krecek pedas",
    "Sop Buntut sapi kuah rempah bening", "Nasi Kuning Manado rica cakalang", "Lontong Sayur Medan tauco labu",
    "Siomay Ikan Tenggiri Bandung", "Tongseng Kambing bumbu rempah", "Ikan Bakar Jimbaran bumbu pedas manis"
]

DRINKS = [
    "Kopi Espresso single origin robusta", "Kopi Susu Gula Aren dingin", "Teh Hijau Matcha latte creamy",
    "Americano dingin tanpa gula", "Jus Alpukat kocok kental saus cokelat", "Teh Earl Grey hangat aroma bergamot",
    "Caffè Latte hangat oat milk", "Air Kelapa muda murni segar", "Wedang Jahe serai hangat gula batu",
    "Es Cincau hitam santan gula kelapa", "Jus Mangga harum manis segar", "Kopi V60 seduh manual biji gayo",
    "Kombucha fermentasi teh apel segar", "Air Lemon madu murni hangat", "Es Teh Melati manis legi",
    "Es Cendol durian dawet ayu", "Jus Sirsak segar selasih"
]

ALLERGIES_DIETS = [
    "alergi makanan laut terutama udang dan kepiting", "alergi kacang tanah dan kacang mete",
    "tidak bisa mengonsumsi makanan bercita rasa pedas", "intoleransi laktosa terhadap produk susu sapi",
    "menjalani pola makan vegetarian murni", "alergi terhadap telur ayam negeri",
    "menjalani diet ketogenik rendah karbohidrat", "alergi gluten pada olahan tepung terigu (celiac)",
    "tidak mengonsumsi daging merah sama sekali", "menjalani diet bebas gula tambahan (sugar free)",
    "pola makan plant-based murni", "riwayat asam lambung (GERD) sehingga membatasi kafein dan asam"
]

PETS = [
    ("Kucing Persia bulu panjang", "Mochi"), ("Kucing Domestik lokal (Oyen)", "Simba"),
    ("Anjing Golden Retriever", "Milo"), ("Kucing British Shorthair abu-abu", "Luna"),
    ("Hamster Roborovski lincah", "Kiko"), ("Kelinci Mini Rex lembut", "Bubu"),
    ("Burung Lovebird warna cerah", "Chirpy"), ("Ikan Cupang hias halfmoon", "Bluey"),
    ("Kucing Munchkin kaki pendek", "Cimol"), ("Anjing Toy Poodle pintar", "Coco"),
    ("Kucing Ragdoll mata biru", "Cleo"), ("Hamster Syrian jinak", "Moci"),
    ("Kucing Scottish Fold telinga lipat", "Oreo"), ("Kura-kura Brazil kecil", "Koko"),
    ("Anjing Shiba Inu setia", "Hachi"), ("Burung Parkit gacor", "Piko")
]

TRAVEL_DESTINATIONS = [
    "Gunung Bromo dan lautan pasir Jawa Timur", "Labuan Bajo dan Taman Nasional Komodo",
    "Ubud pedesaan asri dan Pantai Kuta Bali", "Danau Toba dan Pulau Samosir Sumatera Utara",
    "Kepulauan Raja Ampat surga bawah laut Papua", "Kawah Ijen api biru Banyuwangi",
    "Kawasan Malioboro dan Candi Prambanan Yogyakarta", "Kepulauan Derawan dan Danau Kakaban Kalimantan Timur",
    "Tokyo dan kuil kuno Kyoto Jepang", "Seoul dan keindahan Pulau Jeju Korea Selatan",
    "Kota Wisata Batu dan apel Malang Jawa Timur", "Kemegahan Candi Borobudur Magelang",
    "Gili Trawangan dan pesona Lombok", "Dataran Tinggi Dieng negeri di atas awan Wonosobo",
    "Rumah adat Tongkonan Tana Toraja Sulawesi Selatan", "Pantai Tanjung Tinggi Belitung Laskar Pelangi",
    "Taman Laut Bunaken pesona terumbu karang Manado", "Taman Nasional Wakatobi Sulawesi Tenggara",
    "Bukit Lawang habitat orangutan Sumatera", "Pantai Ora keindahan alam Maluku Tengah"
]

SIDE_BUSINESSES = [
    "kedai kopi susu literan gerobak modern", "jasa pembuatan landing page dan website UMKM",
    "toko pakaian vintage thrift shop online", "produksi dan jualan frozen food risoles mayo rumahan",
    "kursus les privat coding web pemula online", "studio foto mandiri produk katalog e-commerce",
    "jasa desain grafis identitas visual brand", "budidaya tanaman hias hidroponik indoor",
    "jasa titip (jastip) barang hobi impor", "produksi camilan keripik kaca pedas gurih",
    "jasa konsultasi servis hardware laptop dan PC", "agen perjalanan open trip backpacker nusantara",
    "jasa penerjemahan dokumen teknis bahasa Inggris", "toko lilin aromaterapi soy wax handmade",
    "jasa perakitan PC gaming dan custom cooling", "katering menu sehat meal prep harian"
]

FITNESS_GOALS = [
    "program latihan push-pull-legs (PPL) 5 hari seminggu", "latihan endurance persiapan lari maraton 10 kilometer",
    "fokus penambahan massa otot kering (clean bulking)", "rutinitas senam kalisthenics melatih muscle-up dan handstand",
    "latihan kardio interval intensitas tinggi (HIIT)", "peningkatan mobilitas pinggul dan fleksibilitas tubuh",
    "target pencapaian angkatan bench press 100 kg", "latihan berenang jarak menengah untuk stamina jantung",
    "program penurunan persentase lemak tubuh (cutting)", "latihan peregangan yoga vinyasa setiap pagi"
]

CREATIVE_PROJECTS = [
    "menulis naskah novel fiksi ilmiah bertema masa depan AI", "menyusun buku antologi foto jalanan hitam putih",
    "merakit custom keyboard mekanikal dengan switch lubed", "menggambar serial webtoon mingguan bertema komedi kehidupan",
    "membuat channel video tutorial animasi edukasi sains", "merekam mini album musik indie akustik di studio kamar",
    "mendesain set board game strategi bertema sejarah maritim", "membuat diorama miniatur pemandangan stasiun kereta antik"
]

PHILOSOPHY_TOPICS = [
    "prinsip Stoikisme tentang membedakan hal di dalam dan luar kendali diri",
    "misteri horizon peristiwa dan dilatasi waktu di sekitar lubang hitam",
    "dilema etika moral mobil otonom dan regulasi kecerdasan buatan",
    "teori relativitas khusus Einstein tentang elastisitas ruang dan waktu",
    "psikologi pembentukan kebiasaan mikro berbasis sistem (atomic habits)",
    "pandangan eksistensialisme tentang kebebasan dan tanggung jawab hidup",
    "konsep paradoks Fermi mengenai ketiadaan kontak dengan peradaban luar angkasa",
    "hubungan antara ritme sirkadian tubuh dan kejernihan kognitif manusia"
]


# ===========================================================================
# 2. CODE SNIPPETS & DIAGNOSTIC TRACEBACK SCENARIOS
# ===========================================================================

DIAGNOSTIC_SCENARIOS = [
    (
        "Halo asisten! Aplikasiku melempar error berikut di terminal:\n```text\nKeyError: 'user_id'\n  File 'app/services/auth.py', line 45, in authenticate\n    uid = payload['user_id']\n```\nKenapa ini bisa terjadi dan apa perbaikannya?",
        "Error `KeyError: 'user_id'` terjadi karena kunci `'user_id'` tidak ditemukan di dalam dictionary `payload` (misalnya jika token JWT kadaluarsa atau payload belum didecode sempurna).\n\nSolusi terbaik:\n1. **Gunakan method `.get()`** dengan default value atau validasi eksplisit:\n```python\nuid = payload.get('user_id')\nif not uid:\n    raise AuthenticationError('Token tidak memiliki user_id valid')\n```\n2. **Validasi Schema**: Manfaatkan Pydantic atau dataclass untuk memvalidasi payload sebelum diproses."
    ),
    (
        "Tolong bantu, query PostgreSQL ini lambat banget saat tabel mencapai 2 juta baris:\n```sql\nSELECT * FROM orders WHERE customer_id = 9182 ORDER BY created_at DESC LIMIT 20;\n```\nBagaimana cara mengoptimalkannya?",
        "Masalahnya ada pada sorting dan scanning tabel besar tanpa index gabungan. Solusinya buat **Composite Index**:\n```sql\nCREATE INDEX idx_orders_customer_created ON orders (customer_id, created_at DESC);\n```\nDengan index ini, database langsung melakukan *Index Scan* terarah tanpa perlu melakukan *Sort Memory Scan* berbiaya tinggi pada jutaan baris."
    ),
    (
        "Di React aku dapat warning ini di console browser:\n```text\nWarning: Each child in a list should have a unique 'key' prop.\n```\nApakah ini berbahaya dan bagaimana memperbaikinya?",
        "Warning ini muncul saat melakukan render array elemen menggunakan `.map()` tanpa prop `key` unik pada root elemen terluar.\n\nContoh perbaikan:\n```tsx\n// Gunakan ID unik dari data, hindari index jika urutan list bisa berubah\n{items.map((item) => (\n  <div key={item.id} className='p-3 border-b'>\n    <p>{item.name}</p>\n  </div>\n))}\n```\nProp `key` sangat krusial agar algoritma Reconciliation React tahu persis elemen mana yang ditambah, diubah, atau dihapus secara efisien."
    ),
    (
        "Bagaimana cara memecah tugas komputasi asinkron di Python tanpa memblokir event loop?",
        "Gunakan `asyncio.to_thread()` untuk mendelegasikan fungsi I/O sinkron atau fungsi blocking ke thread worker terpisah. Contohnya:\n```python\nimport asyncio\nimport time\n\ndef blocking_io():\n    time.sleep(2)\n    return 'Selesai proses berat'\n\nasync def main():\n    hasil = await asyncio.to_thread(blocking_io)\n    print(hasil)\n```\nDengan cara ini, event loop utama tetap responsif menangani traffic request lainnya."
    ),
    (
        "Bagaimana cara menulis Dockerfile multi-stage build yang ringkas untuk aplikasi Go?",
        "Gunakan stage `builder` dengan image Go lengkap, lalu salin binary hasil compile ke stage `alpine` atau `scratch`:\n```dockerfile\nFROM golang:1.22-alpine AS builder\nWORKDIR /app\nCOPY go.mod go.sum ./\nRUN go mod download\nCOPY . .\nRUN CGO_ENABLED=0 go build -o server .\n\nFROM alpine:latest\nWORKDIR /root/\nCOPY --from=builder /app/server .\nEXPOSE 8080\nCMD [\"./server\"]\n```\nUkuran image akhir bisa terpangkas dari ratusan MB menjadi hanya belasan MB saja."
    )
]


# ===========================================================================
# 3. 120+ EXPANDED INTELLECTUAL & PRACTICAL DISTRACTORS
# ===========================================================================

EXPANDED_DISTRACTORS = [
    # --- Productivity, Cognitive Science & Habits ---
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
    ("Mengapa teknik 'Feynman' sangat efektif untuk mempelajari konsep baru yang rumit?",
     "Teknik Feynman memaksa kita menjelaskan konsep sulit menggunakan bahasa sederhana seolah mengajarkannya kepada anak kecil, sehingga celah pemahaman kita langsung terlihat jelas."),
    ("Apa dampak 'context switching' yang berlebihan terhadap energi mental kita?",
     "Peralihan tugas berulang memicu 'attention residue'—sebagian fokus kognitif kita masih tertinggal di tugas sebelumnya, menurunkan IQ fungsional dan mempercepat kelelahan mental."),

    # --- Software Engineering, Architecture & DevOps ---
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
    ("Apa arti konsep Idempotency dalam perancangan RESTful API?",
     "Operasi dikatakan idempotent jika mengeksekusinya satu kali atau berkali-kali menghasilkan state sistem yang sama persis (contohnya HTTP PUT atau DELETE)."),
    ("Bagaimana cara menangani 'cold start' pada arsitektur Serverless (AWS Lambda)?",
     "Gunakan fitur Provisioned Concurrency, perkecil ukuran bundle deployment, dan pilih runtime berbobot ringan seperti Go, Rust, atau Node.js yang cepat dalam inisialisasi."),

    # --- Health, Fitness, Ergonomics & Nutrition ---
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
    ("Apa peran elektrolit tubuh saat kita berolahraga dalam cuaca yang panas?",
     "Elektrolit seperti natrium, kalium, dan magnesium menjaga keseimbangan cairan sel, mencegah kram otot, dan memastikan transmisi impuls saraf bekerja optimal."),

    # --- Finance, Business & Career Navigation ---
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
     "Compound interest menghitung imbal hasil tidak hanya dari modal awal, tetapi juga dari bunga akumulasi sebelumnya, memicu pertumbuhan eksponensial dalam jangka panjang."),
    ("Apa arti istilah 'Moat' (parit pertahanan) dalam strategi bisnis kompetitif?",
     "Moat adalah keunggulan bersaing berkelanjutan—seperti efek jaringan, brand kuat, paten teknologi, atau switching cost tinggi—yang melindungi bisnis dari gempuran kompetitor."),

    # --- Science, Astronomy, Nature & Physics ---
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
    ("Mengapa gravitasi di bulan jauh lebih kecil dibanding di permukaan bumi?",
     "Karena massa bulan hanya sekitar 1,2% dari massa bumi, sehingga gaya tarikan gravitasinya hanya sekitar seperenam (sekitar 16,6%) gravitasi bumi."),
    ("Bagaimana aurora borealis bisa terbentuk di langit kutub utara?",
     "Aurora tercipta ketika partikel bermuatan dari angin matahari berinteraksi dengan medan magnet bumi dan bertumbukan dengan molekul gas di atmosfer atas."),

    # --- Communication, Philosophy & Culture ---
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


# ===========================================================================
# 4. NATURAL RECALL PHRASING ENGINE (100% IDIOMATIC INDONESIAN)
# ===========================================================================

RECALL_QUESTIONS_POOL: Dict[str, List[str]] = {
    "name": [
        "Kamu masih ingat siapa namaku yang kuperkenalkan tadi?",
        "Bisa sebutkan kembali siapa namaku?",
        "Tadi di awal obrolan, namaku siapa ya?",
        "Coba tes ingatanmu, siapa nama lengkap atau panggilanku?"
    ],
    "job": [
        "Ngomong-ngomong, kamu masih ingat apa profesi pekerjaanku sehari-hari?",
        "Bisa sebutkan profesi pekerjaan yang kuceritakan di obrolan tadi?",
        "Tolong cek memorimu, apa pekerjaanku sehari-hari?",
        "Tadi aku menceritakan berkarir sebagai apa ya?",
        "Kamu ingat nggak bidang pekerjaan utamaku apa?"
    ],
    "city": [
        "Bisa sebutkan di kota mana aku tinggal sekarang?",
        "Kamu masih ingat kota tempat domisili atau tempat tinggalku di mana?",
        "Tadi di perkenalan, di kota mana aku berdomisili?",
        "Aku tinggal dan menetap di kota mana tadi?",
        "Tadi aku menyebutkan tinggal di daerah mana ya?"
    ],
    "lang": [
        "Bahasa pemrograman apa yang tadi kusebutkan sering kupakai?",
        "Kamu masih ingat bahasa koding utama yang kuceritakan tadi?",
        "Tadi aku bilang menggunakan bahasa pemrograman apa di pekerjaanku?",
        "Bahasa software apa yang menjadi andalanku tadi?"
    ],
    "tool": [
        "Framework atau tools teknologi apa yang tadi kuceritakan kupakai?",
        "Kamu ingat framework yang kugunakan dalam proyek kerjaku?",
        "Tadi aku menyebutkan framework apa yang sering kupakai?",
        "Teknologi framework apa yang tadi kubahas di proyekku?"
    ],
    "hobby": [
        "Aktivitas hobi yang biasa kulakukan untuk santai apa tadi?",
        "Hobi yang sering kulakukan untuk melepas lelah tadi apa ya?",
        "Bisa sebutkan kembali kegiatan hobiku yang tadi kuceritakan?",
        "Waktu luangku biasanya kuisi dengan kegiatan apa tadi?",
        "Kegiatan santai favoritku di luar jam kerja tadi apa?"
    ],
    "food": [
        "Makanan favorit yang paling kusukai tadi apa ya?",
        "Kamu ingat jenis hidangan makanan kesukaanku yang kuceritakan?",
        "Tadi aku bilang paling suka menyantap makanan apa?",
        "Menu makanan andalanku yang tadi kusebutkan apa?",
        "Kuliner yang paling membangkitkan seleraku tadi apa ya?"
    ],
    "drink": [
        "Kamu ingat jenis minuman segar yang paling sering kutemani pas santai?",
        "Minuman favorit yang kuceritakan tadi apa ya?",
        "Tadi minuman kesukaanku apa yang kusebutkan?",
        "Minuman penyegar yang tadi kubilang suka kunikmati apa?"
    ],
    "allergy": [
        "Sebelum merekomendasikan makanan, kamu ingat kondisi kesehatan atau pantanganku apa?",
        "Tadi aku menceritakan pantangan makan atau alergi yang kumiliki, masih ingat apa itu?",
        "Bisa sebutkan kondisi alergi atau diet khusus yang harus selalu kuperhatikan?",
        "Pantangan fisik atau riwayat kesehatanku tadi apa yang kusebutkan?"
    ],
    "pet_name": [
        "Siapa nama hewan peliharaan kesayanganku di rumah tadi?",
        "Kamu masih ingat nama peliharaanku yang kusebutkan?",
        "Tolong sebutkan nama dari hewan kesayanganku yang tinggal bersamaku.",
        "Peliharaanku di rumah kuberi nama siapa tadi?"
    ],
    "pet_type": [
        "Jenis hewan peliharaan apa yang kupelihara di rumah tadi?",
        "Kamu ingat jenis binatang peliharaan apa yang kuceritakan tadi?",
        "Hewan apa yang menemaniku di rumah tadi?"
    ],
    "travel": [
        "Tadi destinasi wisata impian yang kurencanakan ke mana ya?",
        "Kamu masih ingat tempat liburan yang ingin kukunjungi tahun ini?",
        "Tadi aku menceritakan rencana jalan-jalan dan liburan ke mana?",
        "Tempat wisata idaman yang kusebutkan tadi di mana?"
    ],
    "side_biz": [
        "Usaha sampingan apa yang sedang kurintis di waktu luang tadi?",
        "Bisnis sampingan apa yang tadi kuceritakan sedang kujalani?",
        "Bisa sebutkan usaha mandiri yang sedang kukembangkan di luar jam kerja?",
        "Project bisnis tambahan yang kurintis tadi apa ya?"
    ],
    "fitness_goal": [
        "Target atau program kebugaran jasmani apa yang sedang kujalani tadi?",
        "Kamu ingat rutinitas latihan fisik yang sedang fokus kulakukan?",
        "Tadi program olahraga atau target kebugaran apa yang sedang kujalankan?",
        "Rencana workout fisik yang kutargetkan tadi apa ya?"
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

RECALL_ANSWERS_POOL: Dict[str, List[str]] = {
    "name": [
        "Tentu saja ingat, namamu adalah {ans}.",
        "Kamu adalah {ans}, senang selalu bisa mendampingimu berdiskusi!",
        "Nama yang kamu perkenalkan di awal percakapan kita adalah {ans}."
    ],
    "job": [
        "Tentu saja, kamu berprofesi dan bekerja sebagai seorang {ans}.",
        "Profesi pekerjaanmu sehari-hari adalah {ans}.",
        "Berdasarkan percakapan kita tadi, kamu berkarir sebagai {ans}.",
        "Tentu, kamu adalah seorang {ans}."
    ],
    "city": [
        "Tentu, kamu berdomisili dan menetap di kota {ans}.",
        "Kota tempat tinggalmu saat ini adalah {ans}.",
        "Di awal percakapan tadi kamu menyebutkan tinggal di kota {ans}.",
        "Kamu tinggal di kawasan kota {ans}."
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
        "Aku selalu mencatatnya dengan cermat: kamu memiliki kondisi {ans}."
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

def format_natural_recall_turn(recall_key: str, ans: str) -> Tuple[str, str]:
    q_cands = RECALL_QUESTIONS_POOL.get(recall_key, [f"Kamu masih ingat {recall_key} yang kuceritakan tadi?"])
    a_cands = RECALL_ANSWERS_POOL.get(recall_key, [f"Tentu saja aku ingat, hal yang kamu ceritakan tadi adalah {ans}."])
    return random.choice(q_cands), random.choice(a_cands).format(ans=ans)


# ===========================================================================
# 5. PERSONA BUILDER
# ===========================================================================

def build_random_persona(uid: str) -> Dict[str, Any]:
    name = random.choice(NAMES)
    city = random.choice(CITIES)
    alt_city = random.choice([c for c in CITIES if c != city])
    job = random.choice(TECH_ROLES if random.random() < 0.52 else NON_TECH_ROLES)
    alt_job = random.choice([j for j in TECH_ROLES + NON_TECH_ROLES if j != job])
    pet_type, pet_name = random.choice(PETS)

    return {
        "uid": uid,
        "name": name,
        "city": city,
        "alt_city": alt_city,
        "job": job,
        "alt_job": alt_job,
        "lang": random.choice(PROGRAMMING_LANGS),
        "tool": random.choice(FRAMEWORKS_AND_TOOLS),
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


# ===========================================================================
# 6. PROCEDURAL DISCOURSE CLAUSE GENERATOR (INFINITE COMBINATIONS)
# ===========================================================================

GREETING_OPENERS = [
    "Halo!", "Hai asisten!", "Selamat pagi rekan AI!", "Selamat siang asisten!",
    "Selamat sore!", "Halo rekan cerdas!", "Permisi mau konsultasi dong!", "Hai apa kabar?",
    "Pagi!", "Halo chatbot!", "Salam hangat!", "Hai halo rekan!", "Permisi rekan AI!"
]

INTRO_CLAUSES = [
    "Kenalkan, namaku {name}. Aku warga {city} yang bekerja sebagai {job}.",
    "Aku {name}, saat ini tinggal di {city} dan berprofesi sebagai {job}.",
    "Salam dari {city}! Panggil aku {name}, sehari-hari aku berkarir sebagai {job}.",
    "Namaku {name}. Sehari-hari aku menetap di kota {city} dengan profesi {job}.",
    "Sebagai seorang {job} yang berdomisili di {city}, perkenalkan aku {name}.",
    "Kenalin aku {name}. Kesibukan utamaku saat ini adalah {job} di kawasan {city}."
]

MID_CONV_PERSONA_DROPS = [
    "Oh iya sekalian kenalan, namaku {name}. Sehari-hari aku berkutat sebagai {job} di kota {city}.",
    "Btw salam kenal ya! Aku {name}, tinggal di {city}. Profesi utamaku itu seorang {job}.",
    "Ngomong-ngomong biar lebih akrab, namaku {name} dari {city}. Sehari-hari aku aktif bekerja sebagai {job}.",
    "Senang berdiskusi denganmu! Kenalkan aku {name}, warga {city} yang berkarir di bidang {job}."
]

ASSISTANT_OPENING_REPLIES = [
    "Halo {name}! Senang sekali bisa menyapamu di {city}. Sebagai seorang {job}, pasti aktivitasmu sangat produktif. Ada topik menarik apa yang ingin kita ulas hari ini?",
    "Hai {name}! Senang bisa berkenalan. Salam hangat untuk seorang {job} dari {city}! Mari kita mulai diskusinya, apa yang bisa kubantu?",
    "Salam kenal {name}! Senang terhubung denganmu. Menarik sekali melihat peranmu sebagai {job} di kota {city}. Silakan ceritakan apa yang ingin kamu diskusikan!",
    "Halo {name} dari {city}! Selamat datang. Profesi sebagai {job} tentu menuntut fokus yang tajam. Ada hal spesifik yang ingin kita bedah bersama?"
]


# ===========================================================================
# 7. 15 DIVERSE DIALOGUE ARCHETYPE BUILDERS
# ===========================================================================

def build_debugging_diagnostic_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """User starts with a real code error / traceback, drops persona in Turn 2, followed by recall."""
    turns, facts = [], []
    q_diag, a_diag = random.choice(DIAGNOSTIC_SCENARIOS)
    turns.append({"role": "user", "content": q_diag})
    turns.append({"role": "assistant", "content": a_diag})

    # Persona in Turn 2
    p_drop = random.choice(MID_CONV_PERSONA_DROPS).format(name=p["name"], city=p["city"], job=p["job"])
    turns.append({"role": "user", "content": f"Penjelasanmu sangat solutif dan jelas! {p_drop} Stack andalanku memang {p['lang']}."})
    turns.append({"role": "assistant", "content": f"Sama-sama {p['name']}! Senang solusinya bermanfaat. Bahasa {p['lang']} memang sangat fleksibel bila penanganan exception dan error handling-nya terstruktur."})
    facts.append({"turn": 2, "key": "name", "value": p["name"]})
    facts.append({"turn": 2, "key": "city", "value": p["city"]})
    facts.append({"turn": 2, "key": "job", "value": p["job"]})
    facts.append({"turn": 2, "key": "lang", "value": p["lang"]})

    # Turn 4: Drink
    turns.append({"role": "user", "content": f"Sehabis pusing berjam-jam cari bug, enaknya langsung istirahat sambil minum {p['drink']} biar rileks."})
    turns.append({"role": "assistant", "content": f"Pilihan istirahat yang sempurna, {p['name']}. Menikmati segelas {p['drink']} ampuh menyegarkan kembali fokus setelah debugging panjang."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    # Distractor turns
    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["job", "city", "lang", "drink"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "debugging_and_diagnostics",
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


def build_culinary_recipe_safety_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Culinary planning with explicit allergy constraint and regional seasonings."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    # Turn 2: Dish inquiry with allergy
    dishes = [
        f"Hari Minggu nanti aku berencana memasak hidangan favoritku, yaitu {p['food']}. Cuma kamu wajib catat ya, aku punya {p['allergy']}. Bagaimana resep modifikasinya?",
        f"Aku kepingin banget masak {p['food']} khas nusantara di rumah, tapi harus steril karena aku {p['allergy']}. Ada ide pengganti bumbu yang aman?",
        f"Menu makanan kesukaanku itu {p['food']}. Tapi karena aku {p['allergy']}, tolong beri saran cara mengolah bumbu alaminya biar tetap gurih dan sedap."
    ]
    turns.append({"role": "user", "content": random.choice(dishes)})
    turns.append({"role": "assistant", "content": (
        f"Tentu, mari kita racik resep {p['food']} yang 100% ramah untuk kondisi {p['allergy']}, {p['name']}!\n"
        f"1. **Substitusi Bumbu**: Ganti bahan pemicu dengan kombinasi kemiri sangrai, bawang merah, bawang putih, dan sedikit ketumbar untuk menghasilkan rasa gurih alami.\n"
        f"2. **Aromatik Segar**: Gunakan serai memar, daun salam, dan daun jeruk segar agar aroma harumnya menutupi kekurangan bahan yang dipantang.\n"
        f"3. **Teknik Menumis**: Tumis bumbu halus dengan api sedang hingga matang sempurna (tanpa langu) sebelum menuangkan kaldu kaldu gurih."
    )})
    facts.append({"turn": 2, "key": "food", "value": p["food"]})
    facts.append({"turn": 2, "key": "allergy", "value": p["allergy"]})

    # Turn 4: Drink
    turns.append({"role": "user", "content": f"Biar santapan kulinernya makin lengkap, minumannya aku siapkan segelas {p['drink']}."})
    turns.append({"role": "assistant", "content": f"Perpaduan yang sangat menggoda selera! Menikmati {p['food']} ditemani segelas {p['drink']} pasti bikin acara makan semakin nikmat."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    # Distractor turns
    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["food", "allergy", "drink"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "culinary_and_allergy_safety",
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


def build_travel_and_exploration_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Travel itinerary, budgeting, and funding via side business."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    travel_queries = [
        f"Salah satu resolusi jalan-jalanku tahun ini adalah menjelajahi keindahan alam di {p['travel']}. Ada saran rencana perjalanan terbaik?",
        f"Tahun ini aku punya target liburan impian ke {p['travel']}. Menurutmu berapa hari durasi yang ideal untuk mengeksplorasi wisatanya?",
        f"Aku lagi merancang liburan healing ke {p['travel']}. Hal apa saja yang wajib dipersiapkan sebelum berangkat?"
    ]
    turns.append({"role": "user", "content": random.choice(travel_queries)})
    turns.append({"role": "assistant", "content": (
        f"Destinasi yang sangat mempesona, {p['name']}! Untuk memaksimalkan perjalanan ke {p['travel']}:\n"
        f"- **Waktu Kunjungan Terbaik**: Datanglah di musim kemarau saat cuaca cerah untuk menikmati pemandangan secara maksimal.\n"
        f"- **Rute Prioritas**: Susun rute harian secara berdekatan agar waktu tidak habis di perjalanan.\n"
        f"- **Perlengkapan**: Siapkan sepatu trekking yang nyaman, jaket tahan angin, dan kamera untuk mengabadikan momen berharga."
    )})
    facts.append({"turn": 2, "key": "travel", "value": p["travel"]})

    # Side biz
    turns.append({"role": "user", "content": f"Biar tabungan jalan-jalanku semakin aman, di luar jam kantor sekarang aku aktif merintis usaha sampingan berupa {p['side_biz']}."})
    turns.append({"role": "assistant", "content": f"Langkah kemandirian finansial yang sangat bijak! Mengembangkan {p['side_biz']} adalah cara cerdas mendanai liburan tanpa mengganggu arus kas utama."})
    facts.append({"turn": 4, "key": "side_biz", "value": p["side_biz"]})

    # Distractor turns
    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["travel", "side_biz", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "travel_and_exploration",
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


def build_fitness_and_periodization_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Fitness regimens, recovery physiology, and hydration."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    turns.append({"role": "user", "content": f"Biar badan tetap sehat dan bugar, saat ini aku lagi fokus berkomitmen menjalani {p['fitness_goal']}. Bagaimana tips menjaga konsistensinya?"})
    turns.append({"role": "assistant", "content": (
        f"Komitmen olahraga yang sangat berharga, {p['name']}! Tiga kunci keberhasilan untuk {p['fitness_goal']}:\n"
        f"1. **Periodisasi Bertahap**: Jangan memaksakan beban atau intensitas maksimal di minggu awal; beri tubuh waktu beradaptasi.\n"
        f"2. **Nutrisi Seimbang**: Konsumsi protein berkualitas (telur, tempe, dada ayam, ikan) untuk sintesis protein otot.\n"
        f"3. **Tidur Pemulihan**: Saat tidur nyenyak hormon pertumbuhan bekerja memperbaiki serat otot yang lelah."
    )})
    facts.append({"turn": 2, "key": "fitness_goal", "value": p["fitness_goal"]})

    turns.append({"role": "user", "content": f"Sehabis sesi workout yang melelahkan, minuman penyegar andalanku adalah segelas {p['drink']}."})
    turns.append({"role": "assistant", "content": f"Pilihan pelepas dahaga yang sangat nikmat! {p['drink']} memberi rasa rileks seketika setelah membakar banyak kalori."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    # Distractor turns
    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["fitness_goal", "drink", "job"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "fitness_and_periodization",
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


def build_casual_storytelling_banter_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Everyday Indonesian life, commute, rain, pets, and relaxing banter."""
    turns, facts = [], []
    anecdotes = [
        (f"Aduh hari ini jalanan di kota {p['city']} macetnya lumayan padat pas jam pulang kantor, baru bisa duduk santai sekarang.",
         f"Pasti cukup menguras tenaga ya. Untung sekarang sudah sampai di rumah dengan aman. Nikmati waktu istirahatmu!"),
        (f"Cuaca sore ini di {p['city']} lagi hujan gerimis syahdu banget, suasana hawanya jadi adem.",
         f"Suasana hujan sejuk memang paling pas untuk rileks sejenak sambil minum teh atau kopi hangat."),
        (f"Hari ini jadwal aktivitasku di {p['city']} cukup padat merayap, tapi bersyukur semuanya tuntas tepat waktu.",
         f"Hebat sekali! Menyelesaikan semua target harian dengan lancar adalah pencapaian yang patut diapresiasi.")
    ]
    u_0, a_0 = random.choice(anecdotes)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})

    name = p["name"]
    pet_lines = [
        f"Untung pas buka pintu rumah langsung disambut tingkah manja {p['pet_type']} peliharaanku si {p['pet_name']}. Btw kenalkan, aku {name}.",
        f"Penawar capek terbaikku ya si {p['pet_name']}, seekor {p['pet_type']} yang selalu bikin gemas. Salam kenal ya, namaku {name}.",
        f"Langsung segar kembali pas main bareng {p['pet_type']} kesayanganku si {p['pet_name']}. Oh iya perkenalkan, namaku {name}."
    ]
    turns.append({"role": "user", "content": random.choice(pet_lines)})
    turns.append({"role": "assistant", "content": f"Halo {name}! Keberadaan {p['pet_type']} bernama {p['pet_name']} memang moodbooster luar biasa setelah seharian menghadapi kepenatan."})
    facts.append({"turn": 2, "key": "name", "value": name})
    facts.append({"turn": 2, "key": "pet_type", "value": p["pet_type"]})
    facts.append({"turn": 2, "key": "pet_name", "value": p["pet_name"]})

    turns.append({"role": "user", "content": f"Di luar urusan rumah, pekerjaanku sehari-hari itu berkarir sebagai seorang {p['job']}."})
    turns.append({"role": "assistant", "content": f"Profesi sebagai {p['job']} pasti membutuhkan dedikasi tinggi. Menjaga kehangatan di rumah bersama {p['pet_name']} adalah harmoni hidup yang sangat indah."})
    facts.append({"turn": 4, "key": "job", "value": p["job"]})

    # Distractor turns
    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["city", "pet_name", "job"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "casual_storytelling_banter",
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


def build_science_and_philosophy_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Deep scientific insights, space exploration, and Stoic philosophy."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    turns.append({"role": "user", "content": f"Akhir-akhir ini aku lagi sering merenungi sebuah topik yang sangat menarik, yaitu {p['philosophy_topic']}. Bagaimana perspektifmu tentang hal ini?"})
    turns.append({"role": "assistant", "content": f"Topik refleksi yang sangat berbobot, {p['name']}! Membedah {p['philosophy_topic']} melatih kejernihan rasionalitas dan membantu kita menavigasi kompleksitas hidup dengan lebih bijak."})
    facts.append({"turn": 2, "key": "philosophy_topic", "value": p["philosophy_topic"]})

    turns.append({"role": "user", "content": f"Membaca dan merenungi topik seperti ini paling nikmat ditemani secangkir {p['drink']} hangat."})
    turns.append({"role": "assistant", "content": f"Suasana yang sangat menenangkan! Kenikmatan {p['drink']} memang partner ideal untuk sesi kontemplasi intelektual."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    # Distractor turns
    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["philosophy_topic", "drink", "job"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
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


def build_creative_and_gaming_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Gaming, digital arts, storytelling, and custom crafting."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    if random.random() < 0.5:
        turns.append({"role": "user", "content": f"Untuk hiburan santai di malam hari, game yang paling sering kumainkan bareng kawan adalah {p['game']}."})
        turns.append({"role": "assistant", "content": f"Pilihan game yang sangat seru, {p['name']}! Game {p['game']} punya gameplay yang memikat dan sangat cocok untuk mencairkan kepenatan."})
        facts.append({"turn": 2, "key": "game", "value": p["game"]})
        target_k = "game"
    else:
        turns.append({"role": "user", "content": f"Di luar jam kantor, saat ini aku lagi asyik menuntaskan proyek kreatif, yaitu {p['creative_project']}."})
        turns.append({"role": "assistant", "content": f"Karya kreasi yang sungguh bernilai seni tinggi, {p['name']}! Menyalurkan energi imajinasi ke dalam {p['creative_project']} adalah cara luar biasa mengekspresikan diri."})
        facts.append({"turn": 2, "key": "creative_project", "value": p["creative_project"]})
        target_k = "creative_project"

    turns.append({"role": "user", "content": f"Selain itu di akhir pekan waktu luangku biasanya kuisi dengan kegiatan {p['hobby']}."})
    turns.append({"role": "assistant", "content": f"Kombinasi aktivitas yang berimbang! Mengisi hari libur dengan {p['hobby']} membuat hidupmu semakin dinamis dan kaya warna."})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    # Distractor turns
    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice([target_k, "hobby", "job"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "creative_and_gaming",
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


def build_memory_update_and_correction_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Explicit fact update or correction scenario."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    update_type = random.choice(["city", "job"])
    if update_type == "city":
        update_lines = [
            f"Kabar terbarunya, bulan depan aku resmi merelokasi tempat tinggal ke kota {p['alt_city']} karena urusan keluarga.",
            f"Oh iya, ada update penting: minggu depan aku mulai pindah rumah dan berdomisili di {p['alt_city']}.",
            f"Sekadar pembaruan info tempat tinggalku, sekarang aku sudah resmi menetap di kota {p['alt_city']}."
        ]
        turns.append({"role": "user", "content": random.choice(update_lines)})
        turns.append({"role": "assistant", "content": f"Selamat atas rencana kepindahan barumu ke kota {p['alt_city']}, {p['name']}! Semoga suasana dan lingkungan baru membawa keberkahan dan kelancaran."})
        facts.append({"turn": 4, "key": "city_updated", "value": p["alt_city"]})
        target_key = "city_updated"
        old_val = p["city"]
        new_val = p["alt_city"]
    else:
        update_lines = [
            f"Kabar gembiranya, per hari ini aku baru saja resmi dipromosikan dan berganti peran menjadi {p['alt_job']}.",
            f"Ada perkembangan positif soal karirku: mulai bulan ini aku mengemban jabatan baru sebagai {p['alt_job']}.",
            f"Info update seputar karirku, aku baru saja menandatangani kontrak peran baru sebagai seorang {p['alt_job']}."
        ]
        turns.append({"role": "user", "content": random.choice(update_lines)})
        turns.append({"role": "assistant", "content": f"Wah selamat banyak atas pencapaian karir barumu sebagai {p['alt_job']}, {p['name']}! Ini tonggak prestasi yang membanggakan."})
        facts.append({"turn": 4, "key": "job_updated", "value": p["alt_job"]})
        target_key = "job_updated"
        old_val = p["job"]
        new_val = p["alt_job"]

    d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
    turns.append({"role": "user", "content": d2[0]})
    turns.append({"role": "assistant", "content": d2[1]})

    if turns_count >= 12:
        d3 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] not in (d1[0], d2[0])])
        turns.append({"role": "user", "content": d3[0]})
        turns.append({"role": "assistant", "content": d3[1]})

    q_rec, a_rec = format_natural_recall_turn(target_key, new_val)
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


def build_math_and_logic_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Mathematical reasoning, logic puzzles, probability, and finance calculations."""
    turns, facts = [], []
    math_challenges = [
        (
            "Tolong bantu hitung: Sebuah toko memberi diskon bertingkat 20% lalu ditambah diskon member 10%. Jika harga awal baju Rp 250.000, berapa harga akhir yang harus dibayar?",
            "Mari kita hitung secara bertahap:\n1. **Diskon Pertama (20%)**: Rp 250.000 × 20% = Rp 50.000. Harga menjadi Rp 200.000.\n2. **Diskon Member (10%)**: Dihitung dari harga setelah diskon pertama: Rp 200.000 × 10% = Rp 20.000.\n3. **Harga Akhir**: Rp 200.000 - Rp 20.000 = **Rp 180.000**.\n\n*Catatan*: Diskon 20% + 10% tidak sama dengan 30% langsung (diskon efektifnya adalah 28%)."
        ),
        (
            "Ada teka-teki logika: Di sebuah ruangan ada 3 sakelar lampu (A, B, C) di luar, dan 1 lampu pijar di dalam ruangan tertutup. Kita hanya boleh masuk ke ruangan satu kali untuk memastikan sakelar mana yang menyalakan lampu. Bagaimana caranya?",
            "Caranya memanfaatkan sifat panas lampu pijar:\n1. Nyalakan sakelar A selama 5-10 menit hingga lampu cukup panas, lalu matikan sakelar A.\n2. Segera nyalakan sakelar B, lalu langsung masuk ke dalam ruangan.\n3. **Analisis kondisi lampu**:\n   - Jika lampu menyala terang -> Sakelar B.\n   - Jika lampu mati tetapi bohlam terasa hangat/panas saat disentuh -> Sakelar A.\n   - Jika lampu mati dan bohlam terasa dingin -> Sakelar C."
        ),
        (
            "Bagaimana cara menghitung Return on Investment (ROI) jika modal investasi awal Rp 15.000.000 dan setelah satu tahun nilai asetnya menjadi Rp 18.750.000?",
            "Rumus ROI adalah: `((Nilai Akhir - Modal Awal) / Modal Awal) × 100%`.\n\nPerhitungannya:\n1. Keuntungan bersih: Rp 18.750.000 - Rp 15.000.000 = Rp 3.750.000.\n2. Rasio keuntungan: Rp 3.750.000 / Rp 15.000.000 = 0,25.\n3. Persentase ROI: 0,25 × 100% = **25% per tahun**.\n\nIni merupakan kinerja imbal hasil investasi yang sangat impresif."
        )
    ]
    q_math, a_math = random.choice(math_challenges)
    turns.append({"role": "user", "content": q_math})
    turns.append({"role": "assistant", "content": a_math})

    name = p["name"]
    turns.append({"role": "user", "content": f"Penjelasan logikanya sangat runtut dan mudah dipahami! Kenalkan namaku {name}, aku berdomisili di {p['city']} dan bekerja sebagai {p['job']}."})
    turns.append({"role": "assistant", "content": f"Salam kenal hangat, {name}! Senang bisa membantu seorang {p['job']} dari kota {p['city']}. Pendekatan matematis analitis seperti ini memang sangat berguna untuk mengambil keputusan rasional sehari-hari."})
    facts.append({"turn": 2, "key": "name", "value": name})
    facts.append({"turn": 2, "key": "city", "value": p["city"]})
    facts.append({"turn": 2, "key": "job", "value": p["job"]})

    turns.append({"role": "user", "content": f"Sambil memecahkan hitung-hitungan logika, biasanya aku rileks menyeruput {p['drink']} favoritku."})
    turns.append({"role": "assistant", "content": f"Suasana yang sangat menyenangkan! Kehadiran {p['drink']} memang teman terbaik untuk menjaga konsentrasi saat berpikir analitis."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["job", "city", "drink"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "math_and_logic_reasoning",
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


def build_linux_devops_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Linux terminal commands, system administration, and infrastructure operations."""
    turns, facts = [], []
    linux_queries = [
        (
            "Bagaimana cara menemukan file berukuran di atas 500MB di Linux yang menghabiskan ruang disk server?",
            "Gunakan perintah `find` yang dikombinasikan dengan format human-readable:\n```bash\nfind / -type f -size +500M -exec ls -lh {} + 2>/dev/null | awk '{ print $5, $9 }'\n```\nAtau gunakan `ncdu /` (NCurses Disk Usage) untuk penelusuran direktori secara interaktif."
        ),
        (
            "Bagaimana perintah systemctl untuk mengecek log error aplikasi yang crash secara realtime?",
            "Gunakan `journalctl` dengan flag `-u` (unit) dan `-f` (follow):\n```bash\nsudo journalctl -u nama-service -f -n 50 --no-pager\n```\nTambahkan flag `-p err` jika kamu hanya ingin memfilter log dengan level error ke atas."
        ),
        (
            "Apa bedanya izin chmod 755 dan chmod 644 pada file dan direktori Linux?",
            "- **chmod 755 (`rwxr-xr-x`)**: Pemilik file punya izin Read, Write, Execute. Group dan pengguna lain hanya punya izin Read dan Execute. Standar untuk folder dan binary eksekusi.\n- **chmod 644 (`rw-r--r--`)**: Pemilik file punya izin Read dan Write. Group dan pengguna lain hanya Read. Standar aman untuk dokumen dan file konfigurasi web server."
        )
    ]
    q_lin, a_lin = random.choice(linux_queries)
    turns.append({"role": "user", "content": q_lin})
    turns.append({"role": "assistant", "content": a_lin})

    conn = random.choice(MID_CONV_PERSONA_DROPS).format(name=p["name"], city=p["city"], job=p["job"])
    turns.append({"role": "user", "content": f"Perintahnya langsung bekerja dengan baik di server! {conn} Di pekerjaan ini kami sering mengelola service berbasis {p['lang']}."})
    turns.append({"role": "assistant", "content": f"Mantap sekali {p['name']}! Senang bisa membantu. Sebagai {p['job']} di {p['city']}, menguasai troubleshooting Linux sangat vital untuk stabilitas environment aplikasi {p['lang']}."})
    facts.append({"turn": 2, "key": "name", "value": p["name"]})
    facts.append({"turn": 2, "key": "city", "value": p["city"]})
    facts.append({"turn": 2, "key": "job", "value": p["job"]})
    facts.append({"turn": 2, "key": "lang", "value": p["lang"]})

    turns.append({"role": "user", "content": f"Kalau lagi standby maintenance server malam hari, cemilan wajibanku itu sepiring {p['food']} hangat."})
    turns.append({"role": "assistant", "content": f"Cemilan pengganjal lapar yang mantap! Santapan {p['food']} pasti ampuh mengusir rasa kantuk saat bertugas."})
    facts.append({"turn": 4, "key": "food", "value": p["food"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["job", "city", "lang", "food"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "linux_and_devops",
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


def build_business_and_startup_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Startup methodology, CAC/LTV, MVP validation, and product marketing."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    biz_questions = [
        (
            "Bagaimana cara memvalidasi ide produk baru dengan biaya minimum sebelum membuat aplikasi lengkapnya?",
            "Gunakan pendekatan **Pre-totyping dan Smoke Test**:\n1. Buat landing page satu halaman yang menjelaskan value proposition produk secara tajam.\n2. Pasang tombol 'Daftar Minat' atau 'Pre-Order' untuk mengukur konversi pengunjung.\n3. Jalankan iklan terarah kecil-kecilan (Rp 100-200 ribu) untuk menguji apakah ada minat pasar nyata sebelum menulis satu baris kode pun."
        ),
        (
            "Apa yang dimaksud dengan rasio LTV / CAC dalam bisnis dan berapa angka yang sehat?",
            "LTV (Customer Lifetime Value) adalah total estimasi profit dari satu pelanggan selama menggunakan produk kita, sedangkan CAC (Customer Acquisition Cost) adalah biaya pemasaran untuk mendapatkan pelanggan tersebut.\n\n- **Rasio 3:1**: Standar industri startup yang sehat dan sustainable.\n- **Rasio < 1:1**: Bisnis merugi pada setiap pelanggan baru.\n- **Rasio > 5:1**: Bisnis mungkin kurang agresif berinvestasi dalam akuisisi pasar."
        )
    ]
    q_b, a_b = random.choice(biz_questions)
    turns.append({"role": "user", "content": q_b})
    turns.append({"role": "assistant", "content": a_b})

    turns.append({"role": "user", "content": f"Prinsip bisnis ini mau kuadopsi untuk mengembangkan usaha sampinganku, yaitu {p['side_biz']}."})
    turns.append({"role": "assistant", "content": f"Aplikasi yang sangat relevan, {p['name']}! Menerapkan validasi pasar pada {p['side_biz']} akan meminimalisir risiko modal dan mempercepat pertumbuhan pelanggan setia."})
    facts.append({"turn": 4, "key": "side_biz", "value": p["side_biz"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["side_biz", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "business_and_startup",
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


def build_english_language_learning_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Linguistics, English grammar, translation nuances, and idiomatic expressions."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    lang_topics = [
        (
            "Bisa jelaskan perbedaan penggunaan Present Perfect Tense ('I have lost my key') vs Simple Past Tense ('I lost my key')?",
            "Perbedaannya ada pada **relevansi dengan masa sekarang (present result)**:\n- **Simple Past** (*I lost my key*): Menjelaskan peristiwa di masa lalu sebagai fakta sejarah. Kunci tersebut mungkin sekarang sudah ditemukan kembali.\n- **Present Perfect** (*I have lost my key*): Menjelaskan bahwa kejadian masa lalu berdampak langsung ke saat ini (artinya: saat ini saya masih belum bisa masuk ke rumah karena kuncinya masih hilang)."
        ),
        (
            "Apa arti ungkapan idiom bahasa Inggris 'Hit the nail on the head' dan bagaimana contoh kalimatnya?",
            "Idiom **'Hit the nail on the head'** artinya: **Tepat sasaran** atau mendeskripsikan situasi/masalah secara persis dan akurat tanpa meleset.\n\nContoh kalimat:\n*\"During the retrospective meeting, Sarah really hit the nail on the head regarding our bottleneck.\"* (Saat meeting evaluasi, Sarah benar-benar tepat sasaran mengenai masalah hambatan kita)."
        )
    ]
    q_l, a_l = random.choice(lang_topics)
    turns.append({"role": "user", "content": q_l})
    turns.append({"role": "assistant", "content": a_l})

    turns.append({"role": "user", "content": f"Belajar bahasa asing memang jadi salah satu hobi santai favoritku, selain aktif {p['hobby']}."})
    turns.append({"role": "assistant", "content": f"Luar biasa, {p['name']}! Memadukan rasa ingin tahu linguistik dengan aktivitas {p['hobby']} adalah latihan stimulasi otak yang sangat positif."})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["hobby", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "language_and_linguistics",
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


def build_office_communication_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Professional workplace communication, email drafting, and polite correspondence."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    turns.append({"role": "user", "content": "Bisa bantu buatkan draf email formal yang sopan untuk mengajukan permohonan cuti tahunan kepada atasan selama 3 hari kerja?"})
    turns.append({"role": "assistant", "content": (
        "Tentu, ini draf email formal yang profesional dan santun:\n\n"
        "**Subjek**: Permohonan Cuti Tahunan - [Nama Lengkap]\n\n"
        "Yth. Bapak/Ibu [Nama Atasan],\n\n"
        "Semoga Bapak/Ibu senantiasa dalam keadaan sehat.\n\n"
        "Melalui email ini, saya bermaksud mengajukan permohonan cuti tahunan selama 3 (tiga) hari kerja, terhitung mulai tanggal [Tanggal Mulai] hingga [Tanggal Selesai].\n\n"
        "Sebelum masa cuti dimulai, saya telah memastikan seluruh tanggung jawab utama dan tugas berjalan telah diselesaikan. Untuk urusan mendesak yang memerlukan koordinasi, rekan saya [Nama Rekan Backup] siap membantu, dan saya tetap dapat dihubungi melalui pesan singkat bila ada keperluan darurat.\n\n"
        "Terima kasih atas perhatian dan persetujuan yang Bapak/Ibu berikan.\n\n"
        "Hormat saya,\n[Nama Lengkap]\n[Posisi/Jabatan]"
    )})

    turns.append({"role": "user", "content": f"Pas waktu istirahat kantor siang tadi, aku makan siang dengan menu kesukaanku yaitu {p['food']}."})
    turns.append({"role": "assistant", "content": f"Menu makan siang yang lezat dan berenergi! Santap hidangan {p['food']} pasti mengembalikan stamina untuk menyelesaikan sisa pekerjaan."})
    facts.append({"turn": 4, "key": "food", "value": p["food"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["food", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "workplace_and_office_communication",
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


def build_indonesian_history_and_culture_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Indonesian history, philosophy of traditional arts, and cultural heritage."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    history_topics = [
        (
            "Apa makna filosofis di balik motif Batik Parang Rusak dalam kebudayaan Jawa?",
            "Motif **Batik Parang Rusak** memiliki simbol garis lengkung diagonal menyerupai ombak karang samudra yang tidak pernah putus. Makna filosofisnya adalah **pantang menyerah, keteguhan hati, dan pengendalian hawa nafsu** dalam menghadapi gelombang rintangan kehidupan."
        ),
        (
            "Bagaimana peran jalur rempah maritim Nusantara dalam membentuk peradaban perdagangan dunia kuno?",
            "Jalur rempah Nusantara (terutama cengkih dan pala dari Maluku) telah menghubungkan kepulauan Indonesia dengan India, Timur Tengah, dan Eropa sejak abad pertama Masehi. Perdagangan maritim ini melahirkan kota-kota pelabuhan kosmopolitan seperti Sriwijaya, Malaka, dan Banten."
        )
    ]
    q_h, a_h = random.choice(history_topics)
    turns.append({"role": "user", "content": q_h})
    turns.append({"role": "assistant", "content": a_h})

    turns.append({"role": "user", "content": f"Bicara soal kekayaan nusantara, salah satu tempat liburan impian yang ingin kukunjungi adalah {p['travel']}."})
    turns.append({"role": "assistant", "content": f"Pilihan destinasi yang sarat pesona! Mengunjungi {p['travel']} pasti memberikan perpaduan indahnya bentang alam dan nilai budaya yang luhur."})
    facts.append({"turn": 4, "key": "travel", "value": p["travel"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["travel", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "history_and_culture",
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


def build_home_diy_and_maintenance_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Practical home maintenance, plant care, and everyday DIY troubleshooting."""
    turns, facts = [], []
    diy_tips = [
        (
            "Ada tips praktis cara mengatasi saluran pipa wastafel dapur yang mulai mampet karena sisa lemak masakan?",
            "Gunakan metode alami tanpa bahan kimia keras:\n1. Tuangkan 1 cangkir **baking soda** (soda kue) kering langsung ke lubang saluran.\n2. Susul dengan 1 cangkir **cuka makan** (akan terjadi reaksi berbusa aktif yang meluruhkan kerak minyak).\n3. Diamkan selama 20-30 menit, lalu siram dengan 1 teko **air panas mendidih** untuk membilas lemak yang sudah larut."
        ),
        (
            "Bagaimana cara merawat tanaman monstera indoor agar daunnya tidak menguning dan tetap rimbun?",
            "Penyebab utama daun menguning adalah *overwatering* (kelebihan air):\n- Siram hanya jika media tanam 2-3 cm bagian atas sudah terasa kering saat disentuh jarimu.\n- Tempatkan di dekat jendela dengan **cahaya terang tidak langsung** (bright indirect light).\n- Pastikan pot memiliki lubang drainase yang lancar agar akar tidak busuk."
        )
    ]
    q_diy, a_diy = random.choice(diy_tips)
    turns.append({"role": "user", "content": q_diy})
    turns.append({"role": "assistant", "content": a_diy})

    p_drop = random.choice(MID_CONV_PERSONA_DROPS).format(name=p["name"], city=p["city"], job=p["job"])
    turns.append({"role": "user", "content": f"Tipsnya sangat aplikatif, langsung kupraktikkan! {p_drop}"})
    turns.append({"role": "assistant", "content": f"Senang sekali bisa membantu, {p['name']}! Sebagai seorang {p['job']} di {p['city']}, merawat hunian agar selalu rapi dan nyaman adalah kunci menjaga ketenangan pikiran."})
    facts.append({"turn": 2, "key": "name", "value": p["name"]})
    facts.append({"turn": 2, "key": "city", "value": p["city"]})
    facts.append({"turn": 2, "key": "job", "value": p["job"]})

    turns.append({"role": "user", "content": f"Di rumah aku juga ditemani peliharaan kesayanganku, yaitu seekor {p['pet_type']} bernama {p['pet_name']}."})
    turns.append({"role": "assistant", "content": f"Luar biasa menggemaskan! Kehadiran {p['pet_type']} bernama {p['pet_name']} pasti selalu bikin suasana rumah semakin ceria."})
    facts.append({"turn": 4, "key": "pet_type", "value": p["pet_type"]})
    facts.append({"turn": 4, "key": "pet_name", "value": p["pet_name"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["pet_name", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "home_maintenance_and_diy",
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


def build_psychology_and_mindset_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Cognitive behavioral insights, imposter syndrome, and psychological resilience."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    psy_topics = [
        (
            "Bagaimana cara mengatasi sindrom imposter (merasa tidak kompeten dan takut dianggap penipu) di lingkungan karir profesional?",
            "Sindrom imposter sangat umum dialami orang-orang berdedikasi tinggi. Tiga langkah mengatasinya:\n1. **Dokumentasikan Fakta Nyata**: Catat pencapaian, apresiasi klien/tim, dan masalah riil yang berhasil kamu selesaikan.\n2. **Ubah Mindset Sempurna**: Mengakui bahwa 'tidak tahu segalanya' adalah hal wajar; yang terpenting adalah kemampuan belajar dan beradaptasi.\n3. **Bicara dengan Mentor**: Berbagi rasa ragu dengan rekan senior akan menyadarkanmu bahwa hampir semua profesional pernah mengalami fase yang sama."
        ),
        (
            "Apa itu 'Cognitive Reframing' dan bagaimana melatihnya saat menghadapi masalah berat?",
            "Cognitive Reframing adalah teknik psikologi untuk mengubah cara kita memandang situasi negatif menjadi perspektif konstruktif. Daripada berpikir 'Ini bencana yang menghancurkan hariku', ubah menjadi 'Ini situasi menantang yang menguji ketahanan dan mengajarkan strategi baru bagi diriku'."
        )
    ]
    q_p, a_p = random.choice(psy_topics)
    turns.append({"role": "user", "content": q_p})
    turns.append({"role": "assistant", "content": a_p})

    turns.append({"role": "user", "content": f"Biar pikiranku tetap tenang dan fokus, biasanya aku meluangkan waktu sejenak untuk {p['hobby']}."})
    turns.append({"role": "assistant", "content": f"Metode grounding yang sangat sehat, {p['name']}. Melakukan {p['hobby']} memicu pelepasan endorfin dan memberikan ruang jeda bagi pikiran untuk kembali seimbang."})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["hobby", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "psychology_and_mindset",
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


def build_ai_and_machine_learning_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Deep learning, Transformer attention, embedding vectors, and model training concepts."""
    turns, facts = [], []
    ai_topics = [
        (
            "Bagaimana mekanisme Self-Attention pada model Transformer memungkinkan pemahaman konteks jarak jauh?",
            "Self-Attention menghitung matriks relevansi antara setiap token dengan seluruh token lainnya menggunakan vektor Query (Q), Key (K), dan Value (V):\n$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\nDengan operasi perkalian dot product ini, jarak posisi antar kata tidak lagi menjadi hambatan (kompleksitas path O(1)), memungkinkan model menangkap ketergantungan semantik jarak jauh secara instan dibanding arsitektur RNN sequential."
        ),
        (
            "Mengapa fungsi Cosine Similarity lebih sering digunakan untuk pencarian semantik vektor dibanding Euclidean Distance?",
            "Cosine Similarity mengukur **sudut kemiringan arah** antar dua vektor, bukan panjang absolut (magnitude)-nya:\n$$\\cos(\\theta) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|}$$\nDalam representasi teks (embedding), panjang vektor sering kali dipengaruhi oleh panjang kalimat atau frekuensi kata, sedangkan makna semantik murni ditentukan oleh orientasi arah vektor dalam ruang laten."
        )
    ]
    q_ai, a_ai = random.choice(ai_topics)
    turns.append({"role": "user", "content": q_ai})
    turns.append({"role": "assistant", "content": a_ai})

    name = p["name"]
    turns.append({"role": "user", "content": f"Penjelasannya sangat mendalam dan presisi! Kenalkan namaku {name}, seorang {p['job']} yang berdomisili di kota {p['city']}."})
    turns.append({"role": "assistant", "content": f"Salam kenal hangat {name}! Senang sekali berdiskusi dengan seorang {p['job']} dari {p['city']}. Konsep representasi vektor ini memang fondasi fundamental di era kecerdasan buatan modern."})
    facts.append({"turn": 2, "key": "name", "value": name})
    facts.append({"turn": 2, "key": "city", "value": p["city"]})
    facts.append({"turn": 2, "key": "job", "value": p["job"]})

    turns.append({"role": "user", "content": f"Di proyek machine learning-ku, bahasa pemrograman yang paling sering kuandalkan adalah {p['lang']}."})
    turns.append({"role": "assistant", "content": f"Pilihan ekosistem yang sangat ideal, {name}. Bahasa {p['lang']} memiliki dukungan pustaka komputasi numerik dan akselerasi hardware yang sangat matang."})
    facts.append({"turn": 4, "key": "lang", "value": p["lang"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["job", "city", "lang"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "ai_and_machine_learning",
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


def build_personal_finance_and_investing_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Personal wealth building, asset allocation, and emergency reserve funds."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    fin_topics = [
        (
            "Berapa idealnya besaran dana darurat yang harus disiapkan untuk seorang profesional lajang dibanding yang sudah berkeluarga?",
            "Besaran dana darurat dihitung berdasarkan pengeluaran rutin bulanan:\n- **Lajang**: Minimal 3 hingga 6 bulan pengeluaran rutin.\n- **Menikah tanpa anak**: 6 hingga 9 bulan pengeluaran rutin.\n- **Menikah dengan anak / Freelancer**: 9 hingga 12 bulan pengeluaran rutin.\n\nSimpan dana darurat di instrumen likuid bebas risiko pasar seperti deposito jangka pendek atau reksa dana pasar uang (RDPU)."
        ),
        (
            "Apa perbedaan metode pelunasan utang 'Debt Snowball' vs 'Debt Avalanche'?",
            "- **Debt Snowball**: Melunasi utang dari nominal saldo terkecil terlebih dahulu, memberi kemenangan psikologis cepat yang membangun momentum disiplin.\n- **Debt Avalanche**: Melunasi utang dari bunga persentase tertinggi terlebih dahulu, secara matematis meminimalkan total beban bunga yang harus dibayarkan."
        )
    ]
    q_f, a_f = random.choice(fin_topics)
    turns.append({"role": "user", "content": q_f})
    turns.append({"role": "assistant", "content": a_f})

    turns.append({"role": "user", "content": f"Selain pekerjaan utama, arus kas tambahanku juga ditopang oleh usaha sampingan berupa {p['side_biz']}."})
    turns.append({"role": "assistant", "content": f"Diversifikasi pendapatan yang sangat sehat, {p['name']}! Memiliki pemasukan dari {p['side_biz']} mempercepat tercapainya tujuan dana darurat dan kemandirian finansial."})
    facts.append({"turn": 4, "key": "side_biz", "value": p["side_biz"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["side_biz", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "personal_finance_and_investing",
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


def build_cinema_and_filmmaking_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Cinematic storytelling, screenwriting structure, and visual grammar."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    film_topics = [
        (
            "Apa fungsi dramatik dari 'Inciting Incident' dalam struktur tiga babak (Three-Act Structure) skenario film?",
            "**Inciting Incident** (Kejadian Pemicu) terjadi di pertengahan Babak Pertama (sekitar menit ke-10 hingga 15), yang merusak keseimbangan dunia normal tokoh utama dan memaksanya keluar dari zona nyaman untuk memulai petualangan cerita."
        ),
        (
            "Bagaimana teknik pencahayaan 'Three-Point Lighting' membentuk kedalaman visual pada sinematografi?",
            "Teknik ini memadukan tiga sumber cahaya:\n1. **Key Light**: Sumber cahaya primer terkuat yang menerangi subjek utama.\n2. **Fill Light**: Cahaya sekunder yang lebih lembut untuk mengisi dan melembutkan bayangan gelap.\n3. **Back Light (Rim Light)**: Cahaya dari belakang subjek untuk memisahkan garis siluet tubuh dari latar belakang, menciptakan kedalaman 3 dimensi."
        )
    ]
    q_c, a_c = random.choice(film_topics)
    turns.append({"role": "user", "content": q_c})
    turns.append({"role": "assistant", "content": a_c})

    turns.append({"role": "user", "content": f"Apresiasi terhadap karya film ini juga memicu proyek kreatifku sendiri, yaitu {p['creative_project']}."})
    turns.append({"role": "assistant", "content": f"Eksplorasi yang sangat kaya estetika, {p['name']}! Menerapkan prinsip naratif visual ke dalam {p['creative_project']} akan membuat karya senimu semakin memikat penikmatnya."})
    facts.append({"turn": 4, "key": "creative_project", "value": p["creative_project"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["creative_project", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "cinema_and_filmmaking",
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


def build_music_theory_and_instruments_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Acoustic guitars, harmony, chord progressions, and audio acoustics."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    music_topics = [
        (
            "Mengapa progresi akor I - V - vi - IV (seperti C - G - Am - F) sangat populer dan adiktif dalam lagu pop dunia?",
            "Progresi ini menciptakan **siklus resolusi emosional sempurna**:\n- **I (Tonic)**: Fondasi rasa aman dan stabil.\n- **V (Dominant)**: Menciptakan ketegangan ekspektasi.\n- **vi (Minor Submediant)**: Menghadirkan sentuhan melankolis reflektif.\n- **IV (Subdominant)**: Transisi hangat yang membawa pendengar bersiap kembali ke nada dasar (resolusi tonic)."
        ),
        (
            "Apa itu 'Circle of Fifths' (Lingkaran Nada Kelima) dan bagaimana musisi memanfaatkannya?",
            "Circle of Fifths memetakan 12 nada tangga nada kromatik berjarak interval 5 nada sempurna. Musisi menggunakannya untuk:\n1. Mengetahui jumlah tanda kres (#) atau mol (b) pada suatu tangga nada secara instan.\n2. Menemukan modulasi akor yang harmonis dan perpindahan nada dasar yang mulus.\n3. Menyusun progresi akor jazz dan harmoni vokal bertingkat."
        )
    ]
    q_m, a_m = random.choice(music_topics)
    turns.append({"role": "user", "content": q_m})
    turns.append({"role": "assistant", "content": a_m})

    turns.append({"role": "user", "content": f"Bermain musik dan nada memang aktivitas pelepas lelah favoritku, selain aktif {p['hobby']}."})
    turns.append({"role": "assistant", "content": f"Keseimbangan rekreasi yang sangat menyehatkan jiwa, {p['name']}! Memadukan harmoni nada dengan kegiatan {p['hobby']} membuat waktu istirahatmu sangat berkualitas."})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["hobby", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "music_and_harmony",
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


def build_gardening_and_agriculture_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Urban hydroponics, plant nutrition, and biological pest prevention."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    garden_topics = [
        (
            "Apa kelebihan sistem hidroponik NFT (Nutrient Film Technique) dibanding sistem wick sumbu untuk sayuran daun?",
            "Sistem NFT mengalirkan larutan nutrisi tipis (film) secara terus-menerus menggunakan pompa kecil:\n- **Oksigenasi Maksimal**: Sebagian akar terendam nutrisi, sebagian akar terekspos udara bebas sehingga respirasi akar sangat optimal.\n- **Pertumbuhan Cepat**: Penyerapan nutrisi berjalan aktif, sayuran seperti selada atau pakcoy dapat dipanen 20-30% lebih cepat dibanding sistem statis."
        ),
        (
            "Bagaimana cara membuat insektisida nabati alami dari minyak mimba (neem oil) untuk membasmi kutu putih (mealybugs)?",
            "Campurkan 1 liter air hangat dengan 5 ml pure cold-pressed neem oil dan 2-3 tetes sabun cuci piring lembut (sebagai emulsifier pengikat minyak). Kocok merata lalu semprotkan ke bawah permukaan daun pada sore hari saat tidak terpapar terik matahari."
        )
    ]
    q_g, a_g = random.choice(garden_topics)
    turns.append({"role": "user", "content": q_g})
    turns.append({"role": "assistant", "content": a_g})

    turns.append({"role": "user", "content": f"Kalau lagi bersantai merawat kebun kecil di rumah, minumannya paling pas segelas {p['drink']} dingin."})
    turns.append({"role": "assistant", "content": f"Momen healing yang sangat menyegarkan! Menikmati suasana hijau ditemani {p['drink']} adalah penawar kepenatan yang sempurna."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["drink", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "gardening_and_agriculture",
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


def build_automotive_and_mechanics_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Automotive mechanics, engine oils, and vehicle maintenance diagnostics."""
    turns, facts = [], []
    auto_topics = [
        (
            "Apa arti kode viskositas oli mesin SAE 10W-40 dan kapan sebaiknya oli diganti?",
            "- **Angka '10W' (Winter)**: Mengukur kekentalan oli pada suhu dingin ekstrem saat mesin pertama kali dihidupkan.\n- **Angka '40'**: Mengukur ketahanan viskositas lapisan oli saat mesin mencapai temperatur kerja panas (100°C).\n- **Interval Ganti**: Pada motor matic/bebek dianjurkan tiap 2.000-3.000 km, sedangkan pada mobil modern tiap 5.000-10.000 km atau maksimal 6 bulan."
        ),
        (
            "Kenapa rem cakram kendaraan bisa berbunyi mendecit tajam saat diinjak?",
            "Penyebab utamanya:\n1. Kampas rem (brake pads) sudah menipis hingga plat indikator logam bergesekan dengan piringan cakram.\n2. Ada debu residu pengereman atau partikel pasir yang terselip di celah kampas.\n3. Piringan cakram bergelombang tidak rata atau mengalami glazing (hangus licin akibat panas pengereman berlebih)."
        )
    ]
    q_a, a_a = random.choice(auto_topics)
    turns.append({"role": "user", "content": q_a})
    turns.append({"role": "assistant", "content": a_a})

    p_drop = random.choice(MID_CONV_PERSONA_DROPS).format(name=p["name"], city=p["city"], job=p["job"])
    turns.append({"role": "user", "content": f"Penjelasan mekanisnya sangat detail! {p_drop}"})
    turns.append({"role": "assistant", "content": f"Senang bisa berbagi info praktis, {p['name']}! Sebagai seorang {p['job']} di {p['city']}, memahami kondisi kendaraan menjamin perjalanan harianmu selalu aman dan lancar."})
    facts.append({"turn": 2, "key": "name", "value": p["name"]})
    facts.append({"turn": 2, "key": "city", "value": p["city"]})
    facts.append({"turn": 2, "key": "job", "value": p["job"]})

    turns.append({"role": "user", "content": f"Biar badan tetap rileks sehabis berurusan dengan mesin dan oli, santap siang kesukaanku itu {p['food']}."})
    turns.append({"role": "assistant", "content": f"Pilihan makanan yang sangat mengenyangkan! Menikmati {p['food']} pasti mengembalikan energimu dengan cepat."})
    facts.append({"turn": 4, "key": "food", "value": p["food"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["food", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "automotive_and_mechanics",
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


def build_cybersecurity_and_privacy_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Information security, encryption hashing, zero-trust architecture, and 2FA."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} {random.choice(INTRO_CLAUSES).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_OPENING_REPLIES).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    sec_topics = [
        (
            "Mengapa fungsi hashing seperti SHA-256 tidak boleh digunakan langsung untuk menyimpan password pengguna tanpa salt dan key stretching?",
            "Karena SHA-256 dirancang sangat cepat untuk integritas data, sehingga rentan diserang menggunakan **Rainbow Tables** dan serangan brute-force hardware GPU (miliaran hash/detik).\n\nGunakan algoritma *Slow Hashing* adaptif yang tahan ASIC/GPU:\n- **Bcrypt**: Menggunakan cost factor komputasi yang bisa dinaikkan berkala.\n- **Argon2id**: Standar pemenang kompetisi hashing modern yang membatasi memori dan resisten serangan side-channel."
        ),
        (
            "Bagaimana prinsip kerja autentikasi Two-Factor Authentication (2FA) berbasis TOTP (Google Authenticator)?",
            "TOTP (Time-based One-Time Password) menggunakan **kunci rahasia bersama (Shared Secret)** yang dibagikan sekali saat scan QR code, digabung dengan **waktu Unix saat ini dibagi per 30 detik** melalui algoritma HMAC-SHA1. Keduanya menghasilkan 6 digit kode unik yang sama persis di ponsel dan server tanpa perlu koneksi internet atau SMS."
        )
    ]
    q_s, a_s = random.choice(sec_topics)
    turns.append({"role": "user", "content": q_s})
    turns.append({"role": "assistant", "content": a_s})

    turns.append({"role": "user", "content": f"Untuk urusan koding sistem keamanan dan backend, stack andalanku adalah {p['lang']}."})
    turns.append({"role": "assistant", "content": f"Pilihan stack yang sangat solid, {p['name']}. Ekosistem {p['lang']} menyediakan pustaka kriptografi standar industri yang mempermudah implementasi secure coding."})
    facts.append({"turn": 4, "key": "lang", "value": p["lang"]})

    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    recall_key = random.choice(["lang", "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "cybersecurity_and_privacy",
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


def build_multi_fact_compound_recall_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Drops 3-4 facts across multiple turns and challenges compound multi-slot memory retrieval."""
    turns, facts = [], []
    g = random.choice(GREETING_OPENERS)
    u_0 = f"{g} Kenalkan, namaku {p['name']}. Aku menetap di kota {p['city']} dan sehari-hari bekerja sebagai {p['job']}."
    a_0 = f"Salam kenal hangat {p['name']}! Senang sekali bisa terhubung dengan seorang {p['job']} dari kota {p['city']}. Apa kabar hari ini?"
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    # Turn 2: Pet drop
    turns.append({"role": "user", "content": f"Di rumah aku memelihara seekor {p['pet_type']} yang kuberi nama {p['pet_name']}."})
    turns.append({"role": "assistant", "content": f"Pasti lucu sekali! Memiliki {p['pet_type']} bernama {p['pet_name']} selalu menghadirkan keceriaan di rumah."})
    facts.append({"turn": 2, "key": "pet_type", "value": p["pet_type"]})
    facts.append({"turn": 2, "key": "pet_name", "value": p["pet_name"]})

    # Turn 4: Drink and hobby drop
    turns.append({"role": "user", "content": f"Kalau lagi senggang di akhir pekan, aktivitasku biasanya {p['hobby']} sambil menikmati segelas {p['drink']}."})
    turns.append({"role": "assistant", "content": f"Kombinasi waktu santai yang sangat menyenangkan! Menikmati {p['drink']} sembari {p['hobby']} adalah cara tepat mengisi ulang energi."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    # Distractor 1
    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    # Compound memory query: testing 2 facts at once!
    pair = random.choice([("city", "pet_name"), ("job", "drink"), ("pet_name", "hobby"), ("city", "job")])
    k1, k2 = pair
    v1, v2 = p[k1], p[k2]
    
    compound_questions = {
        ("city", "pet_name"): f"Kamu masih ingat kota tempat tinggalku dan siapa nama hewan peliharaanku?",
        ("job", "drink"): f"Coba sebutkan profesi pekerjaanku dan jenis minuman favorit santaimu yang kuceritakan tadi?",
        ("pet_name", "hobby"): f"Bisa sebutkan nama hewan peliharaanku serta hobi yang sering kulakukan di akhir pekan?",
        ("city", "job"): f"Bisa ingatkan kembali kota tempat tinggalku dan apa profesi pekerjaanku sehari-hari?"
    }
    compound_answers = {
        ("city", "pet_name"): f"Tentu ingat! Kamu bertempat tinggal di kota {v1}, dan hewan peliharaan kesayanganmu bernama {v2}.",
        ("job", "drink"): f"Tentu saja! Kamu berkarir sebagai seorang {v1}, dan minuman santai favoritmu adalah {v2}.",
        ("pet_name", "hobby"): f"Tentu! Nama peliharaanmu adalah {v1}, dan aktivitas hobi favoritmu adalah {v2}.",
        ("city", "job"): f"Tentu saja! Kamu berdomisili di {v1}, dan profesi pekerjaanmu adalah sebagai {v2}."
    }

    q_rec = compound_questions[pair]
    a_rec = compound_answers[pair]
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "multi_fact_compound_recall",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": f"{k1}+{k2}",
            "ground_truth": f"{v1} & {v2}",
            "question": q_rec,
            "answer": a_rec
        }
    }


def generate_dynamic_math_qa() -> Tuple[str, str]:
    """Generates unbounded, randomized mathematical reasoning questions and step-by-step proofs."""
    problem_type = random.choice(["discount", "roi", "speed_distance", "split_bill", "work_rate"])
    if problem_type == "discount":
        price = random.randint(50, 950) * 1000
        d1 = random.choice([10, 15, 20, 25, 30, 40, 50])
        d2 = random.choice([5, 10, 15, 20])
        p1 = price * (100 - d1) // 100
        final_price = p1 * (100 - d2) // 100
        q = f"Sebuah barang memiliki harga label Rp {price:,}. Toko memberikan promo diskon {d1}%, lalu terdapat diskon tambahan khusus member sebesar {d2}%. Berapa total harga akhir yang harus dibayarkan?".replace(",", ".")
        a = (f"Mari kita hitung secara bertahap:\n"
             f"1. **Diskon Pertama ({d1}%)**: Rp {price:,} × {d1}% = Rp {price*d1//100:,}. Harga menjadi Rp {p1:,}.\n"
             f"2. **Diskon Tambahan ({d2}%)**: Dihitung dari harga baru: Rp {p1:,} × {d2}% = Rp {p1*d2//100:,}.\n"
             f"3. **Harga Akhir**: Rp {p1:,} - Rp {p1*d2//100:,} = **Rp {final_price:,}**.").replace(",", ".")
        return q, a
    elif problem_type == "roi":
        modal = random.randint(5, 50) * 1_000_000
        roi_pct = random.randint(12, 65)
        profit = modal * roi_pct // 100
        akhir = modal + profit
        q = f"Jika seseorang menanamkan modal awal sebesar Rp {modal:,} dalam proyek usaha, dan setelah 1 tahun nilai investasinya menjadi Rp {akhir:,}, berapa persentase ROI (Return on Investment)-nya?".replace(",", ".")
        a = (f"Rumus ROI adalah: `((Nilai Akhir - Modal Awal) / Modal Awal) × 100%`.\n"
             f"1. Keuntungan bersih: Rp {akhir:,} - Rp {modal:,} = Rp {profit:,}.\n"
             f"2. Rasio keuntungan: Rp {profit:,} / Rp {modal:,} = {roi_pct/100:.2f}.\n"
             f"3. Persentase ROI: **{roi_pct}%** per tahun.").replace(",", ".")
        return q, a
    elif problem_type == "speed_distance":
        speed = random.choice([50, 60, 70, 80, 90])
        hours = random.choice([2, 3, 4, 5])
        dist = speed * hours
        city_a = random.choice(["Bandung", "Semarang", "Surabaya", "Malang", "Solo", "Yogyakarta"])
        city_b = random.choice(["Jakarta", "Cirebon", "Madiun", "Banyuwangi", "Tegal", "Kediri"])
        q = f"Sebuah mobil melaju dengan kecepatan rata-rata konstan {speed} km/jam dari {city_a} menuju {city_b} dan membutuhkan waktu perjalanan {hours} jam tanpa henti. Berapa total jarak tempuh kedua kota tersebut?"
        a = f"Total jarak tempuh dihitung dengan rumus `Jarak = Kecepatan × Waktu`:\n$$\\text{{Jarak}} = {speed} \\text{{ km/jam}} \\times {hours} \\text{{ jam}} = **{dist} \\text{{ km}}**$$\nJadi jarak tempuh antara {city_a} dan {city_b} adalah {dist} km."
        return q, a
    elif problem_type == "split_bill":
        n_people = random.randint(3, 7)
        total_makan = random.randint(15, 80) * 10_000
        pajak = total_makan * 10 // 100
        service = total_makan * 5 // 100
        grand_total = total_makan + pajak + service
        per_person = grand_total // n_people
        q = f"Total tagihan makan di restoran adalah Rp {total_makan:,} sebelum pajak (PB1 10%) dan service charge 5%. Jika biaya akhir dibagi rata (split bill) untuk {n_people} orang, berapa rupiah yang harus dibayar masing-masing orang?".replace(",", ".")
        a = (f"Perhitungan pembagian tagihan:\n"
             f"1. **Pajak Restoran (10%)**: Rp {pajak:,}.\n"
             f"2. **Service Charge (5%)**: Rp {service:,}.\n"
             f"3. **Grand Total**: Rp {total_makan:,} + Rp {pajak:,} + Rp {service:,} = Rp {grand_total:,}.\n"
             f"4. **Per Orang ({n_people} orang)**: Rp {grand_total:,} ÷ {n_people} = **Rp {per_person:,}** per orang.").replace(",", ".")
        return q, a
    else:
        p1 = random.choice([4, 6, 8])
        p2 = random.choice([12, 16, 24])
        total_time = (p1 * p2) / (p1 + p2)
        q = f"Pekerja A dapat menyelesaikan suatu proyek dalam {p1} hari, sedangkan Pekerja B dapat menyelesaikannya dalam {p2} hari. Jika keduanya bekerja bersama secara serentak, berapa hari proyek tersebut akan selesai?"
        a = f"Gunakan rumus laju kerja bersama: `1/T = 1/A + 1/B`.\n1. Laju A = 1/{p1} proyek/hari, Laju B = 1/{p2} proyek/hari.\n2. Laju bersama = 1/{p1} + 1/{p2} = {p1+p2}/({p1*p2}).\n3. Waktu penyelesaian (T) = ({p1} × {p2}) / ({p1} + {p2}) = **{total_time:.1f} hari**."
        return q, a


def generate_dynamic_code_qa() -> Tuple[str, str]:
    """Generates unbounded coding challenges and solutions with dynamic parameters."""
    tech = random.choice(["python_algo", "sql_query", "bash_pipeline", "typescript_func", "docker_opt"])
    if tech == "python_algo":
        arr = [random.randint(1, 50) for _ in range(6)]
        target = random.choice(arr)
        q = f"Bagaimana implementasi algoritma Binary Search di Python untuk mencari elemen nilai {target} pada list terurut `{sorted(arr)}`?"
        a = (f"Berikut implementasi Binary Search standar O(log N) di Python:\n"
             f"```python\ndef binary_search(arr, target):\n"
             f"    low, high = 0, len(arr) - 1\n"
             f"    while low <= high:\n"
             f"        mid = (low + high) // 2\n"
             f"        if arr[mid] == target:\n"
             f"            return mid\n"
             f"        elif arr[mid] < target:\n"
             f"            low = mid + 1\n"
             f"        else:\n"
             f"            high = mid - 1\n"
             f"    return -1\n\n"
             f"data = {sorted(arr)}\n"
             f"idx = binary_search(data, {target})\n"
             f"print(f'Elemen ditemukan pada index: {{idx}}')\n```\n"
             f"Kompleksitas waktunya adalah O(log N) dan memori O(1) karena membagi rentang pencarian menjadi separuh pada setiap iterasi.")
        return q, a
    elif tech == "sql_query":
        table = random.choice(["transactions", "user_logs", "order_items", "sensor_telemetry"])
        col_group = random.choice(["user_id", "category_id", "store_branch", "device_type"])
        col_agg = random.choice(["amount", "duration_seconds", "quantity", "battery_level"])
        threshold = random.randint(3, 15)
        q = f"Bagaimana query SQL untuk mencari `{col_group}` yang memiliki total `{col_agg}` lebih dari {threshold * 1000} pada tabel `{table}`?"
        a = (f"Gunakan klausul `GROUP BY` dipadukan dengan filter `HAVING` untuk agregasi:\n"
             f"```sql\nSELECT\n  {col_group},\n  SUM({col_agg}) AS total_metric,\n  COUNT(*) AS total_records\n"
             f"FROM {table}\n"
             f"GROUP BY {col_group}\n"
             f"HAVING SUM({col_agg}) > {threshold * 1000}\n"
             f"ORDER BY total_metric DESC;\n```\n"
             f"`WHERE` menyaring baris sebelum dikelompokkan, sedangkan `HAVING` menyaring hasil kelompok setelah diagregasikan.")
        return q, a
    elif tech == "bash_pipeline":
        logfile = random.choice(["/var/log/nginx/access.log", "production_app.log", "server_requests.log"])
        http_code = random.choice(["500", "404", "502", "403"])
        q = f"Bagaimana one-liner command Linux Bash untuk menghitung 10 IP address teratas yang paling sering memicu error HTTP {http_code} pada file log `{logfile}`?"
        a = (f"Gunakan kombinasi pipeline standar Unix berikut:\n"
             f"```bash\ngrep ' {http_code} ' {logfile} | awk '{{print $1}}' | sort | uniq -c | sort -nr | head -n 10\n```\n"
             f"Alur eksekusi:\n"
             f"1. `grep`: Memfilter baris yang memuat status code {http_code}.\n"
             f"2. `awk`: Mengekstrak kolom pertama (IP address).\n"
             f"3. `sort | uniq -c`: Menghitung frekuensi kemunculan tiap IP unik.\n"
             f"4. `sort -nr | head -n 10`: Mengurutkan dari jumlah terbanyak dan mengambil 10 besar.")
        return q, a
    elif tech == "typescript_func":
        tname = random.choice(["UserProfile", "OrderSummary", "PaymentPayload", "DeviceTelemetry"])
        q = f"Bagaimana cara membuat generic utility type di TypeScript untuk menjadikan semua field pada `{tname}` bersifat readonly dan optional?"
        a = (f"Gunakan mapped types atau kombinasikan utility type bawaan `Readonly<Partial<T>>`:\n"
             f"```typescript\ntype ImmutablePartial<T> = {{\n  readonly [P in keyof T]?: T[P];\n}};\n\n"
             f"// Penggunaan:\ntype Safe{tname} = ImmutablePartial<{tname}>;\n```\n"
             f"Atau cukup gunakan komposisi standar: `type Safe{tname} = Readonly<Partial<{tname}>>;`.")
        return q, a
    else:
        q = "Bagaimana cara mengecilkan ukuran Docker Image Node.js untuk deployment production?"
        a = ("Lakukan beberapa teknik optimasi:\n"
             "1. **Gunakan Alpine atau Distroless base image**: `FROM node:20-alpine`.\n"
             "2. **Multi-stage build**: Pisahkan build stage (yang butuh devDependencies) dari runtime stage.\n"
             "3. **Gunakan `.dockerignore`**: Kecualikan `node_modules`, `.git`, dan file dokumentasi.\n"
             "4. **Install production only**: Jalankan `npm ci --only=production`.\n"
             "Hasilnya ukuran image dapat turun dari ~1GB menjadi di bawah 120MB.")
        return q, a


def build_unbounded_stochastic_dialogue_flow(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Dynamically synthesizes unbounded dialogues with procedural math, code, or science questions."""
    turns, facts = [], []
    
    # Randomly pick procedural question type
    mode = random.choice(["math", "code", "general"])
    if mode == "math":
        q_proc, a_proc = generate_dynamic_math_qa()
    elif mode == "code":
        q_proc, a_proc = generate_dynamic_code_qa()
    else:
        q_proc, a_proc = random.choice(EXPANDED_DISTRACTORS)

    turns.append({"role": "user", "content": q_proc})
    turns.append({"role": "assistant", "content": a_proc})

    # Persona drop in Turn 2
    name = p["name"]
    intro_replies = [
        f"Makasih banyak penjelasannya! Btw kenalkan, namaku {name}. Aku tinggal di {p['city']} dan bekerja sebagai {p['job']}.",
        f"Solusinya sangat mencerahkan! Salam kenal ya, aku {name} dari {p['city']}, sehari-hari sibuk sebagai {p['job']}.",
        f"Keren banget penjelasannya! Kenalin aku {name}. Profesi utamaku {p['job']} dan saat ini berdomisili di kawasan {p['city']}."
    ]
    turns.append({"role": "user", "content": random.choice(intro_replies)})
    turns.append({"role": "assistant", "content": f"Salam kenal hangat, {name}! Senang sekali bisa berdiskusi dengan seorang {p['job']} dari kota {p['city']}. Ada topik seru apa lagi yang ingin kita eksplorasi?"})
    facts.append({"turn": 2, "key": "name", "value": name})
    facts.append({"turn": 2, "key": "city", "value": p["city"]})
    facts.append({"turn": 2, "key": "job", "value": p["job"]})

    # Turn 4: Pet or drink or food drop
    attr = random.choice(["pet", "drink", "food", "hobby"])
    if attr == "pet":
        turns.append({"role": "user", "content": f"Di rumah aku juga ditemani hewan peliharaanku, yaitu seekor {p['pet_type']} yang kuberi nama {p['pet_name']}."})
        turns.append({"role": "assistant", "content": f"Pasti sangat menggemaskan! Kehadiran {p['pet_type']} bernama {p['pet_name']} tentu membawa suasana ceria dan penawar lelah di rumah."})
        facts.append({"turn": 4, "key": "pet_type", "value": p["pet_type"]})
        facts.append({"turn": 4, "key": "pet_name", "value": p["pet_name"]})
        target_k = "pet_name"
    elif attr == "drink":
        turns.append({"role": "user", "content": f"Pas lagi santai istirahat begini, minuman penyegar favoritku itu {p['drink']}."})
        turns.append({"role": "assistant", "content": f"Pilihan pelepas dahaga yang sangat nikmat! {p['drink']} memberi kesegaran seketika untuk melanjutkan hari."})
        facts.append({"turn": 4, "key": "drink", "value": p["drink"]})
        target_k = "drink"
    elif attr == "food":
        turns.append({"role": "user", "content": f"Kalau urusan kuliner makanan kesukaanku di kota {p['city']}, juaranya tetap {p['food']}."})
        turns.append({"role": "assistant", "content": f"Santapan yang sangat menggugah selera! Menu {p['food']} memang punya cita rasa gurih yang istimewa."})
        facts.append({"turn": 4, "key": "food", "value": p["food"]})
        target_k = "food"
    else:
        turns.append({"role": "user", "content": f"Di luar rutinitas pekerjaan, hobi santai yang rutin kujalani di akhir pekan itu {p['hobby']}."})
        turns.append({"role": "assistant", "content": f"Aktivitas yang sangat positif dan menyegarkan! Luang waktu untuk {p['hobby']} menjaga pikiran tetap seimbang."})
        facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})
        target_k = "hobby"

    # Distractor turn
    d_any = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d_any[0]})
    turns.append({"role": "assistant", "content": d_any[1]})

    if turns_count >= 12:
        d_any2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d_any[0]])
        turns.append({"role": "user", "content": d_any2[0]})
        turns.append({"role": "assistant", "content": d_any2[1]})

    recall_key = random.choice([target_k, "job", "city"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": f"unbounded_stochastic_{mode}",
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


ALL_FLOWS = [
    build_debugging_diagnostic_flow,
    build_culinary_recipe_safety_flow,
    build_travel_and_exploration_flow,
    build_fitness_and_periodization_flow,
    build_casual_storytelling_banter_flow,
    build_science_and_philosophy_flow,
    build_creative_and_gaming_flow,
    build_memory_update_and_correction_flow,
    build_math_and_logic_flow,
    build_linux_devops_flow,
    build_business_and_startup_flow,
    build_english_language_learning_flow,
    build_office_communication_flow,
    build_indonesian_history_and_culture_flow,
    build_home_diy_and_maintenance_flow,
    build_psychology_and_mindset_flow,
    build_ai_and_machine_learning_flow,
    build_personal_finance_and_investing_flow,
    build_cinema_and_filmmaking_flow,
    build_music_theory_and_instruments_flow,
    build_gardening_and_agriculture_flow,
    build_automotive_and_mechanics_flow,
    build_cybersecurity_and_privacy_flow,
    build_multi_fact_compound_recall_flow,
    build_unbounded_stochastic_dialogue_flow,
    build_unbounded_stochastic_dialogue_flow,
    build_unbounded_stochastic_dialogue_flow
]


def format_chatml(turns: List[Dict[str, str]]) -> str:
    lines = []
    for t in turns:
        lines.append(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>")
    return "\n".join(lines)


# ===========================================================================
# 8. HIGH-THROUGHPUT STREAMING ENGINE
# ===========================================================================

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

    print("=" * 76)
    print("   INFINITE-DIVERSITY 100M TOKENS MULTI-TURN CONVERSATIONAL GENERATOR")
    print("=" * 76)
    print(f"  Target Total Tokens   : {train_tokens_target + val_tokens_target + test_tokens_target:,}")
    print(f"  Train Target          : {train_tokens_target:,} tokens (95%)")
    print(f"  Validation Target     : {val_tokens_target:,} tokens (2.5%)")
    print(f"  Test Target           : {test_tokens_target:,} tokens (2.5%)")
    print(f"  Tokenizer             : {tokenizer_path}")
    print(f"  Archetypes Available  : {len(ALL_FLOWS)} diverse flows with dynamic clause synthesis")
    print(f"  Entity Scale          : {len(NAMES)} Names, {len(CITIES)} Cities, {len(TECH_ROLES)+len(NON_TECH_ROLES)} Roles")
    print(f"  Intellectual Pool     : {len(EXPANDED_DISTRACTORS)} broad topics + {len(DIAGNOSTIC_SCENARIOS)} code diagnostics")
    print("=" * 76)

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
                batch_items = []
                batch_chatml = []

                for _ in range(batch_size):
                    eid = f"User_{total_written_conversations + len(batch_items) + 1:07d}"
                    persona = build_random_persona(eid)
                    turns_len = random.choice([6, 8, 10, 12, 14])
                    flow_fn = random.choice(ALL_FLOWS)

                    diag = flow_fn(persona, turns_len)
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

                encodings = tok.encode_batch(batch_chatml)

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
    print("\n" + "=" * 76)
    print("                     ALL SPLITS COMPLETED                      ")
    print("=" * 76)
    print(f"  Total Tokens Generated   : {total_generated_tokens:,}")
    print(f"  Total Conversations      : {total_written_conversations:,}")
    print(f"  Total Wallclock Time     : {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    print(f"  Overall Generation Speed : {total_generated_tokens/total_time:,.0f} tokens/second")
    print("=" * 76)

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
    parser = argparse.ArgumentParser(description="Generate Infinite-Diversity 100M Tokens Multi-Turn Conversational Dataset")
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
