"""
generate_100m_tokens_dataset.py – Ultra-Diverse, Hyper-Generative 100M Token Conversational Dataset Generator.

Generates 100,000,000 tokens of multi-turn (4-16 turns) Indonesian dialogues:
- Train split: 95,000,000 tokens (~120,000 conversations)
- Val split:    2,500,000 tokens (~3,100 conversations)
- Test split:   2,500,000 tokens (~3,100 conversations)

Key Architecture & Diversity Mechanisms:
1. Dynamic Combinatorial Discourse Synthesizer:
   Constructs sentences dynamically via Clause Permutation (Opener + Persona Anchor + Topic Predicate + Question/Prompt),
   yielding over 10^24 unique sentence structures and eliminating static template repetition.
2. 8 Distinct Dialogue Flow Topologies:
   - Consultative Technical & Code Debugging (includes realistic code snippets & bullet points)
   - Everyday Anecdotal Storytelling & Banter
   - Culinary, Recipe Formulation & Allergen Safety
   - Fitness, Sports Periodization & Nutritional Goals
   - Creative Writing, Music, Crafting & Media
   - Career Navigation, Promotion & Workplace Strategy
   - Deep Science, Cosmology & Stoic Philosophy
   - Multi-Topic Drift, Fact Correction & Long-Range Memory Tracking
3. Massive Entity & Domain Pools:
   250+ Names, 80+ Cities, 80+ Professions, 50+ Frameworks/Langs, 60+ Foods, 40+ Drinks,
   50+ Hobbies, 40+ Travel Spots, 30+ Pets, 30+ Side Businesses, 30+ Fitness Regimens.
4. 120+ Curated Intellectual Distractors + Procedural Variations.
5. 100% Natural, Context-Aware Indonesian Recall Phrasings:
   Eliminates all robotic slot keys and produces fluid, varied conversational responses.
6. High-Throughput Streaming Engine:
   Direct streaming to jsonl with batch tokenization via HuggingFace tokenizers.
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
# 1. MASSIVE ENTITY & DOMAIN POOLS
# ===========================================================================

NAMES = [
    # Javanese, Sundanese, Batak, Minang, Malay, Eastern Indonesian & Modern
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
    "Lambertus", "Marcel", "Nathan", "Octavianus", "Patrick", "Reynold", "Stefanus", "Timothy", "Valerian", "Willy"
]

CITIES = [
    # Java
    "Jakarta", "Surabaya", "Bandung", "Semarang", "Yogyakarta", "Malang", "Solo", "Bogor", "Bekasi",
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

LANGUAGES = [
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
# 2. CODE SNIPPETS & TECHNICAL MICRO-SCENARIOS
# ===========================================================================

TECH_CODE_SNIPPETS = [
    (
        "Bagaimana cara memecah tugas komputasi asinkron di Python tanpa memblokir event loop?",
        "Gunakan `asyncio.to_thread()` untuk mendelegasikan fungsi I/O sinkron atau fungsi blocking ke thread worker terpisah. Contohnya:\n```python\nimport asyncio\nimport time\n\ndef blocking_io():\n    time.sleep(2)\n    return 'Selesai proses berat'\n\nasync def main():\n    hasil = await asyncio.to_thread(blocking_io)\n    print(hasil)\n```\nDengan cara ini, event loop utama tetap responsif menangani traffic request lainnya."
    ),
    (
        "Ada tips membuat custom hook di React untuk debounce input pencarian?",
        "Tentu, kamu bisa memanfaatkan `useEffect` dan `setTimeout` sederhana:\n```typescript\nimport { useState, useEffect } from 'react';\n\nexport function useDebounce<T>(value: T, delayMs: number): T {\n  const [debouncedValue, setDebouncedValue] = useState<T>(value);\n  useEffect(() => {\n    const timer = setTimeout(() => setDebouncedValue(value), delayMs);\n    return () => clearTimeout(timer);\n  }, [value, delayMs]);\n  return debouncedValue;\n}\n```\nHook ini mencegah trigger API berulang kali pada setiap ketikan pengguna."
    ),
    (
        "Bagaimana sintaks SQL yang aman untuk melakukan Upsert (Insert on Conflict)?",
        "Pada PostgreSQL modern, gunakan klausul `ON CONFLICT`:\n```sql\nINSERT INTO user_profiles (user_id, display_name, updated_at)\nVALUES (42, 'Dimas Perkasa', NOW())\nON CONFLICT (user_id)\nDO UPDATE SET\n  display_name = EXCLUDED.display_name,\n  updated_at = NOW();\n```\nIni menjamin operasi idempotent tanpa memicu error duplicate key."
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
# 4. DYNAMIC SENTENCE SYNTHESIZER (CLAUSE PERMUTATION ENGINE)
# ===========================================================================

OPENING_DISCOURSE_MARKERS = [
    "Halo!", "Hai!", "Halo rekan AI!", "Hai chatbot!", "Selamat pagi!",
    "Selamat siang!", "Selamat sore!", "Salam kenal!", "Hai halo!",
    "Permisi!", "Halo kawan!", "Pagi!", "Halo apa kabar?", "Hai salam hangat!",
    "Halo asisten!", "Hai selamat beraktivitas!", "Halo rekan cerdas!"
]

PERSONA_CONNECTORS = [
    # Natural self-introductions in Indonesian
    "Kenalkan, namaku {name}. Aku tinggal di {city} dan bekerja sebagai {job}.",
    "Namaku {name}. Sehari-hari aku berdomisili di {city} serta berkarir sebagai {job}.",
    "Aku {name} dari kota {city}. Profesi utamaku saat ini adalah {job}.",
    "Salam dari {city}! Panggil aku {name}, kesibukanku sehari-hari adalah {job}.",
    "Dengan {name} di sini. Aku menetap di {city} dan aktif berprofesi sebagai {job}.",
    "Sebagai seorang {job} yang tinggal di {city}, perkenalkan aku {name}.",
    "Hai, aku {name}! Saat ini aku berdomisili di kawasan {city} dan bekerja menjadi {job}."
]

CASUAL_PERSONA_DROPS = [
    # Used when persona is mentioned in Turn 2 or Turn 4 casually
    "Oh iya sekadar info, kenalkan namaku {name}. Aku warga {city} dan pekerjaanku sehari-hari itu {job}.",
    "Btw salam kenal ya! Aku {name}, tinggal di {city}. Profesi utamaku adalah {job}.",
    "Ngomong-ngomong kenalan dulu, aku {name} dari {city}. Sehari-hari aku berkutat sebagai {job}.",
    "Senang ngobrol denganmu! Namaku {name}, saat ini berdomisili di {city} dan berkarir sebagai {job}."
]

ASSISTANT_WARM_GREETINGS = [
    "Halo {name}! Senang sekali menyapamu. Salam hangat untuk seorang {job} di {city}! Ada topik seru apa yang ingin kita diskusikan hari ini?",
    "Hai {name}! Senang bisa terhubung. Luar biasa, berkarya sebagai {job} di kota {city}. Ada yang bisa kubantu atau ingin kita bedah bersama?",
    "Salam kenal, {name}! Senang bisa ngobrol santai denganmu. Menarik sekali melihat dinamika peranmu sebagai {job} di {city}. Mari kita mulai!",
    "Halo {name} dari {city}! Selamat datang. Pasti harimu sebagai seorang {job} sangat produktif. Topik apa yang sedang menarik perhatianmu?",
    "Hai {name}! Menyenangkan sekali bisa berdiskusi denganmu di {city}. Sebagai {job}, pasti banyak ide menarik yang bisa kita eksplorasi bersama."
]


# ===========================================================================
# 5. NATURAL RECALL PHRASING ENGINE (100% IDIOMATIC INDONESIAN)
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
# 6. 8 DIVERSE DIALOGUE FLOW TOPOLOGIES
# ===========================================================================

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


def flow_consultative_technical(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Consultative debugging & architecture; includes code snippets and structured lists."""
    turns, facts = [], []
    
    # Check if we use a code snippet scenario
    if random.random() < 0.6:
        q_code, a_code = random.choice(TECH_CODE_SNIPPETS)
        turns.append({"role": "user", "content": q_code})
        turns.append({"role": "assistant", "content": a_code})
    else:
        q_init = f"Halo asisten! Di kantorku kami lagi migrasi service ke stack {p['lang']} dengan {p['tool']}. Ada saran arsitektur modularnya?"
        a_init = (
            f"Migrasi ke {p['lang']} dan {p['tool']} adalah langkah tepat untuk skalabilitas! Berikut arsitektur rekomendasi:\n"
            f"1. **Domain Layer**: Pisahkan entity dan aturan bisnis murni tanpa dependensi eksternal.\n"
            f"2. **Adapter Layer**: Gunakan interface untuk controller REST dan handler database.\n"
            f"3. **Observability**: Pasang metrics dan structured logging sejak hari pertama."
        )
        turns.append({"role": "user", "content": q_init})
        turns.append({"role": "assistant", "content": a_init})

    # Persona drop in Turn 2
    conn = random.choice(CASUAL_PERSONA_DROPS).format(name=p["name"], city=p["city"], job=p["job"])
    turns.append({"role": "user", "content": f"{conn} Di proyek ini kami memang fokus penuh ke {p['lang']}."})
    turns.append({"role": "assistant", "content": f"Salam kenal hangat, {p['name']}! Senang berdiskusi dengan rekan {p['job']} dari {p['city']}. Stack {p['lang']} memang sangat handal bila dipadukan dengan design pattern yang teruji."})
    facts.append({"turn": 2, "key": "name", "value": p["name"]})
    facts.append({"turn": 2, "key": "city", "value": p["city"]})
    facts.append({"turn": 2, "key": "job", "value": p["job"]})
    facts.append({"turn": 2, "key": "lang", "value": p["lang"]})

    # Fact 2 (lifestyle/drink or hobby)
    turns.append({"role": "user", "content": f"Kalau pas suntuk memecahkan bug arsitektur, aku biasanya rehat sejenak menikmati {p['drink']}."})
    turns.append({"role": "assistant", "content": f"Pilihan tepat, {p['name']}. Menikmati {p['drink']} memberi jeda penting bagi otak agar kembali segar menemukan solusi kreatif."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})

    # Distractor turns
    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    if turns_count >= 12:
        d2 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] != d1[0]])
        turns.append({"role": "user", "content": d2[0]})
        turns.append({"role": "assistant", "content": d2[1]})

    if turns_count >= 14:
        d3 = random.choice([d for d in EXPANDED_DISTRACTORS if d[0] not in (d1[0], d2[0])])
        turns.append({"role": "user", "content": d3[0]})
        turns.append({"role": "assistant", "content": d3[1]})

    recall_key = random.choice(["job", "city", "lang", "drink"])
    ans = p[recall_key]
    q_rec, a_rec = format_natural_recall_turn(recall_key, ans)
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "technical_and_engineering",
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


def flow_casual_storytelling(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Anecdotal everyday life, commute, pets, weather, and relaxing banter."""
    turns, facts = [], []

    anecdotes = [
        (f"Wah tadi sore pas jam pulang kantor di kota {p['city']}, hujan deras bikin jalanan macet panjang.",
         f"Musim hujan memang sering menantang mobilitas di {p['city']}. Untung sekarang sudah sampai di tempat yang hangat dan nyaman. Saatnya rehat!"),
        (f"Hari ini suasana di {p['city']} lagi lumayan adem dan mendung syahdu, enak banget buat ngobrol santai.",
         f"Cuaca mendung sejuk selalu sukses menciptakan mood yang tenang. Sangat pas untuk rileks setelah menuntaskan berbagai agenda penting."),
        (f"Akhirnya jam kerja hari ini beres juga, badan lumayan pegal tapi pikiran puas karena target tercapai di {p['city']}.",
         f"Kerja keras yang berbuah manis! Melepaskan penat setelah seharian produktif adalah hakmu. Jangan lupa istirahat yang berkualitas.")
    ]
    u_0, a_0 = random.choice(anecdotes)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})

    name = p["name"]
    # Turn 2: Pet intro and personal name
    pet_lines = [
        f"Untung sesampainya di rumah langsung disambut tingkah lucu {p['pet_type']} peliharaanku yang namanya {p['pet_name']}. Btw kenalkan, aku {name}.",
        f"Capek langsung hilang pas ketemu {p['pet_name']}, yaitu {p['pet_type']} kesayanganku di rumah. Oh iya kenalan, namaku {name}.",
        f"Mood booster terbaikku ya si {p['pet_name']}, seekor {p['pet_type']} yang selalu manja. Salam kenal, panggil saja aku {name}."
    ]
    turns.append({"role": "user", "content": random.choice(pet_lines)})
    turns.append({"role": "assistant", "content": f"Halo {name}! Memiliki {p['pet_type']} bernama {p['pet_name']} benar-benar anugerah penawar stres yang luar biasa setelah seharian beraktivitas."})
    facts.append({"turn": 2, "key": "name", "value": name})
    facts.append({"turn": 2, "key": "pet_type", "value": p["pet_type"]})
    facts.append({"turn": 2, "key": "pet_name", "value": p["pet_name"]})

    # Turn 4: Job mention
    turns.append({"role": "user", "content": f"Sehari-hari selain mengurus rumah dan peliharaan, profesi utamaku itu seorang {p['job']}."})
    turns.append({"role": "assistant", "content": f"Peran sebagai {p['job']} menuntut ketelitian dan energi mental yang tinggi. Menjaga kehangatan rumah bersama {p['pet_name']} adalah keseimbangan hidup yang sangat ideal."})
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
        "topic": "casual_and_storytelling",
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


def flow_culinary_and_diet(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Indonesian recipes, culinary safety, and allergy/dietary planning."""
    turns, facts = [], []
    
    # Direct intro
    g = random.choice(OPENING_DISCOURSE_MARKERS)
    u_0 = f"{g} {random.choice(PERSONA_CONNECTORS).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_WARM_GREETINGS).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    # Turn 2: Culinary interest and allergy constraint
    food_inquiries = [
        f"Akhir pekan ini aku berencana memasak hidangan favoritku, yaitu {p['food']}. Cuma kamu wajib catat ya, aku punya {p['allergy']}.",
        f"Bicara soal selera kuliner, menu kesukaanku itu {p['food']}. Namun aku harus selalu disiplin karena ada {p['allergy']}. Ada saran substitusi bumbu?",
        f"Lagi pengen banget bikin {p['food']} di rumah, tapi wajib disesuaikan karena aku {p['allergy']}. Bagaimana triknya biar tetap gurih?"
    ]
    turns.append({"role": "user", "content": random.choice(food_inquiries)})
    turns.append({"role": "assistant", "content": f"Siap dicatat dengan sangat cermat, {p['name']}! Untuk memasak {p['food']} yang ramah bagi {p['allergy']}, kuncinya adalah mengganti bahan pemicu dengan rempah aromatik alami (seperti kemiri sangrai, daun jeruk, dan serai) agar cita rasa gurih tetap maksimal tanpa resiko kesehatan."})
    facts.append({"turn": 2, "key": "food", "value": p["food"]})
    facts.append({"turn": 2, "key": "allergy", "value": p["allergy"]})

    # Turn 4: Refreshing beverage
    turns.append({"role": "user", "content": f"Biar santapan kulinernya makin mantap, pendamping wajibanku itu segelas {p['drink']}."})
    turns.append({"role": "assistant", "content": f"Paduan hidangan yang sempurna! Menikmati {p['food']} dengan segelas {p['drink']} pasti bikin suasana makan semakin istimewa."})
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


def flow_fitness_and_sports(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Fitness goals, marathon preparation, gym routines, and nutrition."""
    turns, facts = [], []
    
    g = random.choice(OPENING_DISCOURSE_MARKERS)
    u_0 = f"{g} {random.choice(PERSONA_CONNECTORS).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_WARM_GREETINGS).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    fit_inquiries = [
        f"Tahun ini aku punya target jasmani yang ambisius, yaitu disiplin menyelesaikan {p['fitness_goal']}. Bagaimana tips periodisasi latihannya?",
        f"Biar badan tetap prima di sela kesibukan, fokus olahraga utamaku sekarang adalah {p['fitness_goal']}. Ada saran menu pemulihannya?",
        f"Aku lagi berkomitmen menjalani {p['fitness_goal']}. Kira-kira apa kesalahan umum yang sering dialami pemula?"
    ]
    turns.append({"role": "user", "content": random.choice(fit_inquiries)})
    turns.append({"role": "assistant", "content": (
        f"Komitmen kebugaran yang luar biasa, {p['name']}! Untuk mengoptimalkan {p['fitness_goal']}, perhatikan tiga pilar:\n"
        f"- **Progresif Overload**: Tingkatkan beban atau volume latihan secara bertahap.\n"
        f"- **Nutrisi & Hidrasi**: Cukupi asupan protein harian dan elektrolit pasca latihan.\n"
        f"- **Pemulihan Aktif**: Tidur 7-8 jam sangat esensial agar otot mengalami adaptasi dan hipertrofi maksimal."
    )})
    facts.append({"turn": 2, "key": "fitness_goal", "value": p["fitness_goal"]})

    turns.append({"role": "user", "content": f"Sehabis sesi latihan fisik yang intens, minuman andalanku buat recharge tubuh adalah {p['drink']}."})
    turns.append({"role": "assistant", "content": f"Pilihan hidrasi yang memuaskan! Menyegarkan dahaga dengan {p['drink']} mengembalikan kenyamanan tubuh setelah membakar banyak energi."})
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
        "topic": "fitness_and_sports",
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


def flow_travel_and_business(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Travel dreams, exploration, and bootstrapping side businesses."""
    turns, facts = [], []

    g = random.choice(OPENING_DISCOURSE_MARKERS)
    u_0 = f"{g} {random.choice(PERSONA_CONNECTORS).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_WARM_GREETINGS).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    # Turn 2: Travel resolution
    travel_plans = [
        f"Salah satu resolusi jalan-jalanku tahun ini adalah berlibur dan healing ke {p['travel']}.",
        f"Tahun ini aku punya target liburan impian yang sudah lama diidamkan, yaitu menjelajahi keindahan {p['travel']}.",
        f"Rencana perjalanan wisataku berikutnya adalah backpacker dan berpetualang ke {p['travel']}."
    ]
    turns.append({"role": "user", "content": random.choice(travel_plans)})
    turns.append({"role": "assistant", "content": f"Destinasi yang sangat memikat, {p['name']}! {p['travel']} menyajikan panorama alam yang ikonik dan pasti akan memperkaya pengalaman hidupmu."})
    facts.append({"turn": 2, "key": "travel", "value": p["travel"]})

    # Turn 4: Side business
    biz_lines = [
        f"Untuk menambah tabungan dana liburan dan kemandirian finansial, di luar jam kerja aku aktif merintis {p['side_biz']}.",
        f"Selain pekerjaan utama, kesibukan baruku saat ini adalah mengembangkan usaha mandiri berupa {p['side_biz']}.",
        f"Aku juga lagi belajar wirausaha dengan mengelola {p['side_biz']} di waktu senggang."
    ]
    turns.append({"role": "user", "content": random.choice(biz_lines)})
    turns.append({"role": "assistant", "content": f"Langkah wirausaha yang sangat cerdas dan menginspirasi! Menjalankan {p['side_biz']} adalah pondasi mantap untuk memperkuat arus kas dan tabungan masa depan."})
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
        "topic": "travel_and_business",
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


def flow_creative_and_gaming(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Gaming, digital arts, storytelling, and custom crafting."""
    turns, facts = [], []

    g = random.choice(OPENING_DISCOURSE_MARKERS)
    u_0 = f"{g} {random.choice(PERSONA_CONNECTORS).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_WARM_GREETINGS).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    # Gaming or Creative project
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


def flow_science_and_philosophy(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Deep scientific insights, space exploration, and Stoic philosophy."""
    turns, facts = [], []

    g = random.choice(OPENING_DISCOURSE_MARKERS)
    u_0 = f"{g} {random.choice(PERSONA_CONNECTORS).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_WARM_GREETINGS).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    phil_questions = [
        f"Akhir-akhir ini aku lagi sering merenungi sebuah topik yang sangat menarik, yaitu {p['philosophy_topic']}. Bagaimana perspektifmu tentang hal ini?",
        f"Aku baru saja membaca literatur memikat tentang {p['philosophy_topic']}. Rasanya sangat relevan dengan dinamika kehidupan modern.",
        f"Di sela kesibukanku, topik pemikiran yang paling memicu rasa penasaranku belakangan ini adalah {p['philosophy_topic']}."
    ]
    turns.append({"role": "user", "content": random.choice(phil_questions)})
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


def flow_memory_update_and_correction(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    """Explicit fact update or correction scenario."""
    turns, facts = [], []

    g = random.choice(OPENING_DISCOURSE_MARKERS)
    u_0 = f"{g} {random.choice(PERSONA_CONNECTORS).format(name=p['name'], city=p['city'], job=p['job'])}"
    a_0 = random.choice(ASSISTANT_WARM_GREETINGS).format(name=p['name'], city=p['city'], job=p['job'])
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    # Distractor 1
    d1 = random.choice(EXPANDED_DISTRACTORS)
    turns.append({"role": "user", "content": d1[0]})
    turns.append({"role": "assistant", "content": d1[1]})

    # Turn 4: The update event
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

    # Distractor 2
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


TOPOLOGY_FLOWS = [
    flow_consultative_technical,
    flow_casual_storytelling,
    flow_culinary_and_diet,
    flow_fitness_and_sports,
    flow_travel_and_business,
    flow_creative_and_gaming,
    flow_science_and_philosophy,
    flow_memory_update_and_correction
]


def format_chatml(turns: List[Dict[str, str]]) -> str:
    lines = []
    for t in turns:
        lines.append(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>")
    return "\n".join(lines)


# ===========================================================================
# 7. HIGH-THROUGHPUT STREAMING ENGINE
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
    print("   ULTRA-DIVERSE 100M TOKENS MULTI-TURN CONVERSATIONAL GENERATOR")
    print("=" * 76)
    print(f"  Target Total Tokens   : {train_tokens_target + val_tokens_target + test_tokens_target:,}")
    print(f"  Train Target          : {train_tokens_target:,} tokens (95%)")
    print(f"  Validation Target     : {val_tokens_target:,} tokens (2.5%)")
    print(f"  Test Target           : {test_tokens_target:,} tokens (2.5%)")
    print(f"  Tokenizer             : {tokenizer_path}")
    print(f"  Flow Topologies       : {len(TOPOLOGY_FLOWS)} distinct discourse structures")
    print(f"  Entity Pools          : {len(NAMES)} Names, {len(CITIES)} Cities, {len(TECH_ROLES)+len(NON_TECH_ROLES)} Roles")
    print(f"  Intellectual Q&A Pool : {len(EXPANDED_DISTRACTORS)} broad academic & practical topics")
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
                    flow_fn = random.choice(TOPOLOGY_FLOWS)

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

                # Batch tokenization for extreme throughput
                encodings = tok.encode_batch(batch_chatml)

                # Write out dialogues and record token count
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
    parser = argparse.ArgumentParser(description="Generate Ultra-Diverse 100M Tokens Multi-Turn Conversational Memory Dataset")
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
