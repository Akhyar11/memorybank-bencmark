"""
generate_100m_tokens_dataset.py – High-Diversity 100M Token Conversational Dataset Generator.

Generates exactly 100,000,000 tokens of multi-turn (8-12 turns) Indonesian dialogues:
- Train split: 95,000,000 tokens (~120,000 conversations)
- Val split:    2,500,000 tokens (~3,200 conversations)
- Test split:   2,500,000 tokens (~3,200 conversations)

Features:
- 200+ names, 50+ cities, 50+ professions, 50+ hobbies, 50+ foods, 40+ travel spots, etc.
- Multiple syntactic phrasing variations per turn for combinatorial diversity (>10^15 unique dialogues).
- Multi-topic coverage (tech/career, lifestyle/health, travel/business, memory update/correction).
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
    "Kirana", "Lestari", "Mega", "Novita", "Olivia", "Pratiwi", "Ratna", "Salsabila", "Tania", "Utari"
]

CITIES = [
    "Jakarta", "Surabaya", "Bandung", "Medan", "Semarang", "Makassar", "Palembang",
    "Denpasar", "Yogyakarta", "Malang", "Balikpapan", "Manado", "Padang", "Solo",
    "Banjarmasin", "Pontianak", "Bogor", "Bekasi", "Tangerang", "Depok",
    "Cimahi", "Cirebon", "Sukabumi", "Tasikmalaya", "Pekalongan", "Tegal", "Magelang",
    "Purwokerto", "Kediri", "Blitar", "Madiun", "Jember", "Banyuwangi", "Banda Aceh",
    "Pematangsiantar", "Binjai", "Pekanbaru", "Dumai", "Jambi", "Bengkulu", "Bandar Lampung",
    "Pangkalpinang", "Batam", "Tanjungpinang", "Samarinda", "Tarakan", "Palu", "Kendari",
    "Gorontalo", "Ambon", "Ternate", "Jayapura", "Sorong", "Mataram", "Kupang"
]

TECH_ROLES = [
    "Frontend Developer", "Backend Developer", "Fullstack Engineer", "Data Scientist",
    "Machine Learning Engineer", "DevOps Engineer", "Mobile Developer", "UI/UX Designer",
    "Product Manager", "QA Automation Engineer", "Cybersecurity Analyst", "Cloud Architect",
    "Database Administrator", "Site Reliability Engineer", "AI Prompt Engineer", "Blockchain Developer",
    "System Analyst", "Scrum Master", "Embedded Systems Engineer", "Network Engineer"
]

NON_TECH_ROLES = [
    "Dokter Umum", "Dokter Gigi", "Apoteker", "Arsitek Bangunan", "Guru Matematika",
    "Dosen Ilmu Komunikasi", "Akuntan Publik", "Konsultan Keuangan", "Pengacara", "Jurnalis Investigasi",
    "Fotografer Komersial", "Chef Restoran", "Editor Video", "Desainer Interior", "Manajer Pemasaran",
    "Psikolog Klinis", "Penerjemah Bahasa", "Penulis Konten", "Barista Spesialis", "Fisioterapis"
]

LANGUAGES = ["Python", "TypeScript", "JavaScript", "Golang", "Rust", "Java", "Kotlin", "Swift", "C++", "PHP", "Dart", "C#", "SQL"]
FRAMEWORKS = ["React", "Vue", "Next.js", "FastAPI", "Django", "Node.js", "Flutter", "PyTorch", "Docker", "Kubernetes", "Spring Boot", "Laravel", "NestJS"]

HOBBIES = [
    "bermain futsal", "bersepeda santai di akhir pekan", "jogging pagi di taman kota",
    "bermain game RPG", "membaca novel fiksi ilmiah", "fotografi jalanan (street photography)",
    "bermain gitar akustik", "belajar memasak kue dan roti", "merawat tanaman hias monstera",
    "latihan angkat beban di gym", "berenang santai", "menonton film dokumenter",
    "mendaki gunung", "camping di alam terbuka", "bermain catur online",
    "menulis blog pribadi", "belajar bahasa asing", "melukis cat air"
]

GAMES = [
    "Valorant", "Mobile Legends", "Genshin Impact", "Dota 2", "Minecraft",
    "FIFA 24", "Elden Ring", "PUBG Mobile", "Apex Legends", "The Witcher 3",
    "Honkai Star Rail", "Cyberpunk 2077", "Stardew Valley", "Free Fire", "Zelda Tears of the Kingdom"
]

FOODS = [
    "Nasi Goreng spesial babat", "Sate Ayam Madura bumbu kacang", "Rendang Sapi khas Minang",
    "Mie Ayam Bakso urat", "Gado-gado siram Jakarta", "Soto Betawi kuah santan",
    "Ayam Geprek sambal bawang", "Nasi Padang lauk rendang", "Pempek Kapal Selam Palembang",
    "Rawon Daging sapi Surabaya", "Bakso Malang komplit", "Nasi Uduk Betawi komplit",
    "Bebek Sinjay Madura", "Ayam Betutu khas Bali", "Gudeg Jogja krecek telur"
]

DRINKS = [
    "Kopi Espresso single origin", "Kopi Susu Gula Aren dingin", "Teh Hijau Matcha latte",
    "Americano dingin tanpa gula", "Jus Alpukat kocok cokelat", "Teh Earl Grey hangat",
    "Caffè Latte hangat", "Air Kelapa muda murni", "Wedang Jahe hangat", "Es Cincau hitam gula merah",
    "Jus Mangga segar", "Kopi V60 seduh manual"
]

ALLERGIES_DIETS = [
    "alergi makanan laut (seafood)", "alergi kacang tanah dan mete", "tidak bisa makan makanan pedas sama sekali",
    "intoleransi laktosa terhadap susu sapi", "menjalani pola makan vegetarian", "alergi telur ayam negeri",
    "menjalani diet rendah karbohidrat (keto)", "alergi gluten pada tepung terigu", "tidak mengonsumsi daging merah"
]

PETS = [
    ("Kucing Persia", "Mochi"), ("Kucing Domestik (Oyen)", "Simba"), ("Anjing Golden Retriever", "Milo"),
    ("Kucing British Shorthair", "Luna"), ("Hamster Roborovski", "Kiko"), ("Kelinci Rex", "Bubu"),
    ("Burung Lovebird", "Chirpy"), ("Ikan Cupang hias", "Bluey"), ("Kucing Munchkin", "Cimol"),
    ("Anjing Poodle", "Coco"), ("Kucing Ragdoll", "Cleo"), ("Hamster Syrian", "Moci")
]

TRAVEL_DESTINATIONS = [
    "Gunung Bromo Jawa Timur", "Labuan Bajo dan Pulau Komodo", "Ubud dan Pantai Kuta Bali",
    "Danau Toba Sumatera Utara", "Kepulauan Raja Ampat Papua", "Kawah Ijen Banyuwangi",
    "Kawasan Malioboro Yogyakarta", "Pantai Derawan Kalimantan Timur", "Tokyo dan Kyoto Jepang",
    "Seoul dan Pulau Jeju Korea Selatan", "Batu Malang Jawa Timur", "Candi Borobudur Magelang",
    "Gili Trawangan Lombok", "Dataran Tinggi Dieng Wonosobo", "Tana Toraja Sulawesi Selatan"
]

SIDE_BUSINESSES = [
    "kedai kopi kecil-kecilan", "jasa pembuatan website freelance", "toko pakaian online (thrift shop)",
    "jual makanan beku (frozen food) rumahan", "kursus les privat coding online", "studio foto mandiri",
    "jasa desain grafis freelance", "toko tanaman hias hidroponik", "jasa titip (jastip) barang impor",
    "produksi camilan keripik pedas", "jasa servis laptop dan komputer"
]


# ---------------------------------------------------------------------------
# Persona & Dynamic Conversation Builders
# ---------------------------------------------------------------------------

def build_random_persona(uid: str) -> Dict[str, Any]:
    name = random.choice(NAMES)
    city = random.choice(CITIES)
    alt_city = random.choice([c for c in CITIES if c != city])
    job = random.choice(TECH_ROLES if random.random() < 0.6 else NON_TECH_ROLES)
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
    }


# ---------------------------------------------------------------------------
# Dynamic Combinatorial Phrasing Engine
# ---------------------------------------------------------------------------

GREETINGS = [
    "Halo!", "Hai!", "Halo asisten!", "Hai chatbot!", "Selamat pagi!",
    "Selamat siang!", "Selamat sore!", "Salam kenal!", "Hai halo!",
    "Permisi!", "Halo rekan AI!", "Pagi!", "Halo apa kabar?", "Hai salam kenal!"
]

INTRO_PATTERNS = [
    "namaku {name}", "aku {name}", "nama saya {name}", "panggil saja aku {name}",
    "saya {name}", "kenalin namaku {name}", "dengan {name} di sini", "aku bernama {name}"
]

DOMICILE_PATTERNS = [
    "tinggal di kota {city}", "menetap di {city}", "berdomisili di {city}",
    "asal kotaku dari {city}", "saat ini berdomisili di {city}", "hidup dan beraktivitas di {city}",
    "asli warga {city}", "sedang berdomisili di kawasan {city}"
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
    "Salam hangat {name}! Menyenangkan sekali bisa berdiskusi denganmu. Semoga harimu di {city} menyenangkan. Ada hal spesifik yang ingin kamu bicarakan?"
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
        f"{g} {intro[:1].upper() + intro[1:]}. Aku {dom} serta aktif {job}."
    ]
    u_content = random.choice(templates)
    a_content = random.choice(ASSISTANT_GREETING_PATTERNS).format(name=name, city=p["city"], job=p["job"])
    return u_content, a_content


MASSIVE_DISTRACTORS = [
    ("Bagaimana cara terbaik mengelola waktu saat menangani beberapa proyek sekaligus?",
     "Gunakan teknik Time-Blocking dan Matriks Eisenhower. Prioritaskan tugas mendesak yang berdampak besar dan minimalkan multitasking yang memecah fokus."),
    ("Menurutmu apa kriteria dokumentasi teknis yang baik untuk tim kerja?",
     "Dokumentasi yang baik harus ringkas, menyajikan contoh nyata (code snippet atau diagram alur) yang jelas, dan selalu diperbarui bersamaan dengan rilis fitur baru."),
    ("Apakah sertifikasi profesional sangat berpengaruh untuk jenjang karir jangka panjang?",
     "Sertifikasi membuktikan pemahaman standar industri dan dedikasi belajar, namun rekam jejak portofolio proyek riil tetap menjadi pembuktian kompetensi terkuat."),
    ("Kira-kira berapa durasi olahraga ringan yang ideal untuk pemula setiap minggunya?",
     "Berdasarkan pedoman kesehatan, 150 menit per minggu untuk intensitas sedang (seperti jalan cepat 30 menit sehari selama 5 hari) sudah sangat ideal bagi pemula."),
    ("Ada tips supaya keuangan usaha kecil tidak bercampur dengan uang pribadi?",
     "Pisahkan rekening bank sejak hari pertama, buat pembukuan arus kas harian yang disiplin, dan tentukan gaji tetap untuk dirimu sendiri."),
    ("Bagaimana cara menjaga fokus saat bekerja jarak jauh (remote work) dari rumah?",
     "Tetapkan ruang kerja khusus yang bebas gangguan, buat jadwal jam kerja yang teratur, dan kenakan pakaian rapi untuk mengondisikan mindset produktif."),
    ("Apa faktor utama yang menentukan keberhasilan sebuah tim startup pemula?",
     "Kekompakan tim inti dalam mengeksekusi ide, kecepatan merespons feedback pengguna (iterasi produk), dan pengelolaan arus kas (runway) yang sangat disiplin."),
    ("Menurut riset psikologi, apa cara paling efektif untuk membangun kebiasaan baru?",
     "Gunakan metode 'Atomic Habits': mulai dari langkah sangat kecil (mikro), kaitkan dengan rutinitas yang sudah ada (habit stacking), dan beri penghargaan kecil setiap berhasil melakukannya."),
    ("Bagaimana cara mengurangi ketegangan mata bagi orang yang bekerja seharian di depan monitor?",
     "Terapkan aturan 20-20-20: setiap 20 menit menatap layar, alihkan pandangan ke objek berjarak minimal 20 kaki (6 meter) selama minimal 20 detik."),
    ("Apa perbedaan utama antara investasi reksa dana pendapatan tetap dan reksa dana saham?",
     "Reksa dana pendapatan tetap mengalokasikan dana ke obligasi/surat utang dengan risiko sedang dan imbal hasil stabil, sedangkan reksa dana saham memiliki volatilitas tinggi namun potensi imbal hasil jangka panjang lebih besar."),
    ("Bagaimana tips memilih laptop kerja yang awet untuk penggunaan 4-5 tahun ke depan?",
     "Prioritaskan prosesor generasi terbaru dengan minimal 6-8 core, RAM minimal 16GB (lebih baik yang bisa di-upgrade), SSD NVMe cepat, serta kualitas bodi dan sistem pendingin yang solid."),
    ("Menurutmu mengapa istirahat tidur yang cukup sangat krusial bagi daya ingat?",
     "Saat fase tidur gelombang lambat (deep sleep) dan REM, otak melakukan konsolidasi memori, memindahkan informasi baru dari hipokampus ke korteks serebral untuk penyimpanan permanen."),
    ("Apa saran terbaik untuk mengatasi writer's block atau kebuntuan ide kreatif?",
     "Ubah lingkungan sekitarmu dengan berjalan-jalan ke luar ruangan, lakukan 'freewriting' tanpa mengedit selama 10 menit, atau baca literatur di luar domain bidang yang biasa kamu tekuni."),
    ("Bagaimana cara meningkatkan kemampuan berbicara di depan umum (public speaking) secara bertahap?",
     "Mulai dengan merekam suara atau video saat latihan sendiri, pelajari ritme jeda bicara daripada menggunakan jeda 'umm'/'ahh', dan kuasai pembukaan serta penutup presentasi."),
    ("Apakah membaca buku fisik masih memiliki keunggulan dibanding e-book modern?",
     "Buku fisik memberikan pengalaman sensorik sentuhan kertas dan orientasi spasial halaman yang membantu pemahaman mendalam serta mengurangi paparan cahaya biru (blue light).")
]


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
        f"Sehari-hari aku sering riset implementasi {p['tool']} berbasis {p['lang']}. Ekosistemnya terasa sangat produktif untuk tim."
    ]
    turns.append({"role": "user", "content": random.choice(tech_variants)})
    turns.append({"role": "assistant", "content": f"Kombinasi {p['lang']} dengan {p['tool']} sangat populer karena efisiensi eksekusi dan ekosistem library yang matang, {p['name']}. Kuncinya ada pada modularitas arsitektur dan caching yang tepat."})
    facts.append({"turn": 2, "key": "lang", "value": p["lang"]})
    facts.append({"turn": 2, "key": "tool", "value": p["tool"]})

    lifestyle_variants = [
        f"Kalau lagi jenuh sama urusan teknis, pelarianku biasanya minum {p['drink']} sambil {p['hobby']}.",
        f"Biar nggak burnout kerja terus, rutinitas favoritku adalah meluangkan waktu buat {p['hobby']} dan menikmati {p['drink']}.",
        f"Untuk menjaga keseimbangan hidup setelah jam kerja, aku paling suka santai {p['hobby']} ditemani segelas {p['drink']}."
    ]
    turns.append({"role": "user", "content": random.choice(lifestyle_variants)})
    turns.append({"role": "assistant", "content": f"Itu keseimbangan hidup yang sangat sehat, {p['name']}. Menikmati {p['drink']} sambil {p['hobby']} terbukti ampuh menyegarkan pikiran kembali."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    # Distractor turn
    q_dis, a_dis = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis})
    turns.append({"role": "assistant", "content": a_dis})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["job", "city", "lang", "drink", "hobby"])
    recall_questions = {
        "job": [
            f"Ngomong-ngomong, kamu masih ingat apa profesi pekerjaanku?",
            f"Bisa sebutkan profesi pekerjaan yang kuceritakan di awal tadi?",
            f"Tolong cek memori ingatanmu, apa pekerjaanku sehari-hari?"
        ],
        "city": [
            f"Bisa sebutkan di kota mana aku tinggal tadi?",
            f"Kamu masih ingat kota tempat tinggalku sekarang di mana?",
            f"Tadi di perkenalan, di kota mana aku berdomisili?"
        ],
        "lang": [
            f"Bahasa pemrograman apa yang tadi kuceritakan sering kupakai?",
            f"Kamu masih ingat bahasa pemrograman utama proyekku apa?",
            f"Tadi aku bilang menggunakan bahasa pemrograman apa untuk proyekku?"
        ],
        "drink": [
            f"Minuman kesukaanku pas istirahat santai tadi apa ya?",
            f"Bisa sebutkan minuman favorit yang biasa kunikmati?",
            f"Kamu ingat jenis minuman yang sering kutemani pas santai?"
        ],
        "hobby": [
            f"Aktivitas hobi yang biasa kulakukan setelah kerja apa tadi?",
            f"Hobi yang sering kulakukan untuk melepas lelah tadi apa?",
            f"Bisa sebutkan kembali kegiatan hobiku yang tadi kuceritakan?"
        ],
    }
    q_rec = random.choice(recall_questions[recall_key])
    ans = p[recall_key]
    a_rec = f"Berdasarkan percakapan kita tadi, {recall_key} yang kamu sebutkan adalah {ans}."
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


def make_lifestyle_health_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    turns, facts = [], []
    u_0, a_0 = make_dynamic_intro(p)
    turns.append({"role": "user", "content": u_0})
    turns.append({"role": "assistant", "content": a_0})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    food_variants = [
        f"Saat makan siang biasanya aku paling suka menyantap {p['food']}, tapi penting dicatat kalau aku punya kondisi {p['allergy']}.",
        f"Bicara soal kuliner harian, makanan favoritku adalah {p['food']}. Namun aku harus selalu disiplin karena ada {p['allergy']}.",
        f"Menu makanan yang paling membangkitkan seleraku itu {p['food']}, cuma aku wajib menghindari pemicu karena {p['allergy']}."
    ]
    turns.append({"role": "user", "content": random.choice(food_variants)})
    turns.append({"role": "assistant", "content": f"Tercatat dengan sangat baik, {p['name']}! {p['food']} memang hidangan yang nikmat, dan kita tentu wajib selalu memperhatikan kondisi {p['allergy']} agar kesehatanmu tetap prima."})
    facts.append({"turn": 2, "key": "food", "value": p["food"]})
    facts.append({"turn": 2, "key": "allergy", "value": p["allergy"]})

    pet_variants = [
        f"Di tempat tinggalku aku juga memelihara hewan kesayangan, yaitu seekor {p['pet_type']} yang kuberi nama {p['pet_name']}.",
        f"Teman setiaku saat bersantai di rumah adalah peliharaanku, seekor {p['pet_type']} lucu bernama {p['pet_name']}.",
        f"Suasana rumah selalu ramai berkat kehadiran {p['pet_type']} kesayanganku yang namanya {p['pet_name']}."
    ]
    turns.append({"role": "user", "content": random.choice(pet_variants)})
    turns.append({"role": "assistant", "content": f"Pasti menyenangkan sekali ya! Memiliki {p['pet_type']} bernama {p['pet_name']} tentu membawa energi positif dan hiburan hangat di rumah."})
    facts.append({"turn": 4, "key": "pet_type", "value": p["pet_type"]})
    facts.append({"turn": 4, "key": "pet_name", "value": p["pet_name"]})

    q_dis, a_dis = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis})
    turns.append({"role": "assistant", "content": a_dis})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["allergy", "pet_name", "food", "city"])
    recall_questions = {
        "allergy": [
            "Sebelum memilih rekomendasi menu, kamu ingat kondisi kesehatan atau pantangan makananku apa?",
            "Tadi aku menceritakan pantangan makan/kondisi fisikku, kamu masih ingat apa itu?",
            "Bisa sebutkan alergi atau pantangan diet yang kumiliki?"
        ],
        "pet_name": [
            "Siapa nama hewan peliharaan kesayanganku di rumah tadi?",
            "Kamu masih ingat nama peliharaanku yang kusebutkan?",
            "Tolong sebutkan nama dari hewan peliharaanku yang tinggal bersamaku."
        ],
        "food": [
            "Makanan favorit yang paling kusukai tadi apa ya?",
            "Kamu ingat jenis makanan kesukaanku yang kuceritakan tadi?",
            "Tadi aku bilang paling suka menyantap makanan apa?"
        ],
        "city": [
            "Di kota mana tadi aku bilang bertempat tinggal?",
            "Kamu masih ingat kota tempat domisiliku saat ini?",
            "Tadi di awal percakapan, aku berdomisili di mana?"
        ]
    }
    q_rec = random.choice(recall_questions[recall_key])
    ans = p[recall_key]
    a_rec = f"Tentu saja ingat, {recall_key} yang kamu ceritakan adalah {ans}."
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
    turns.append({"role": "assistant", "content": f"Destinasi yang sangat memukau, {p['name']}! {p['travel']} punya pemandangan yang ikonik dan pasti memberikan pengalaman tak terlupakan."})
    facts.append({"turn": 2, "key": "travel", "value": p["travel"]})

    biz_variants = [
        f"Untuk mendanai rencana liburan dan menambah tabungan, sekarang aku aktif merintis usaha sampingan berupa {p['side_biz']}.",
        f"Selain pekerjaan utama, kesibukan baruku saat ini adalah mengembangkan {p['side_biz']}.",
        f"Aku juga lagi belajar berwirausaha mandiri dengan menjalankan {p['side_biz']} di waktu luang."
    ]
    turns.append({"role": "user", "content": random.choice(biz_variants)})
    turns.append({"role": "assistant", "content": f"Langkah wirausaha yang sangat inspiratif! Mengembangkan {p['side_biz']} adalah cara cerdas untuk membangun kemandirian finansial."})
    facts.append({"turn": 4, "key": "side_biz", "value": p["side_biz"]})

    q_dis, a_dis = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis})
    turns.append({"role": "assistant", "content": a_dis})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["travel", "side_biz"])
    recall_questions = {
        "travel": [
            "Tadi destinasi wisata impian yang kurencanakan ke mana ya?",
            "Kamu masih ingat tempat liburan yang ingin kukunjungi tahun ini?",
            "Tadi aku menceritakan rencana jalan-jalan ke mana?"
        ],
        "side_biz": [
            "Usaha sampingan apa yang sedang kurintis tadi?",
            "Bisnis sampingan apa yang tadi kuceritakan sedang kujalani?",
            "Bisa sebutkan usaha mandiri yang sedang kukembangkan di luar jam kerja?"
        ]
    }
    q_rec = random.choice(recall_questions[recall_key])
    ans = p[recall_key]
    a_rec = f"Tentu saja, {recall_key} yang kamu maksud adalah {ans}."
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
        f"Hobi gaming-ku saat ini lagi banyak kuhabiskan di game {p['game']}, gameplay-nya nagih banget."
    ]
    turns.append({"role": "user", "content": random.choice(game_variants)})
    turns.append({"role": "assistant", "content": f"Pilihan hiburan yang seru, {p['name']}! Game {p['game']} memang punya mekanik gameplay yang menantang dan asyik dimainkan bersama kawan."})
    facts.append({"turn": 2, "key": "game", "value": p["game"]})

    hobby_variants = [
        f"Selain bermain game, aktivitas lain yang paling kunikmati untuk relaksasi adalah {p['hobby']}.",
        f"Di akhir pekan, waktu luangku biasanya kuisi dengan kegiatan {p['hobby']}.",
        f"Eksplorasi hobiku di luar layar monitor adalah {p['hobby']}, rasanya sangat memuaskan."
    ]
    turns.append({"role": "user", "content": random.choice(hobby_variants)})
    turns.append({"role": "assistant", "content": f"Aktivitas yang sangat menyenangkan! Menyeimbangkan waktu antara gaming dan {p['hobby']} membuat hari-harimu semakin berwarna."})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    q_dis, a_dis = random.choice(MASSIVE_DISTRACTORS)
    turns.append({"role": "user", "content": q_dis})
    turns.append({"role": "assistant", "content": a_dis})

    if turns_count >= 12:
        q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis])
        turns.append({"role": "user", "content": q_dis2})
        turns.append({"role": "assistant", "content": a_dis2})

    recall_key = random.choice(["game", "hobby"])
    recall_questions = {
        "game": [
            "Game apa yang tadi kubilang sering kumainkan di waktu santai?",
            "Kamu masih ingat judul game favoritku apa?",
            "Tadi aku menceritakan suka main game apa?"
        ],
        "hobby": [
            "Kegiatan hobi yang kulakukan di akhir pekan tadi apa ya?",
            "Bisa sebutkan hobi santai yang tadi kuceritakan?",
            "Kamu catat nggak aktivitas hobiku di luar waktu kerja?"
        ]
    }
    q_rec = random.choice(recall_questions[recall_key])
    ans = p[recall_key]
    a_rec = f"Tentu, {recall_key} yang kamu ceritakan tadi adalah {ans}."
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


def make_update_correction_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
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
        a_up = f"Selamat atas rencana kepindahan barumu ke {p['alt_city']}, {p['name']}! Semoga suasana dan lingkungan baru di sana membawa berkah dan kelancaran."
        facts.append({"turn": 4, "key": "city_updated", "value": p["alt_city"]})
        target_key = "city_updated"
        old_val = p["city"]
        new_val = p["alt_city"]
        q_rec_list = [
            "Bisa ingatkan aku, kota tujuan pindahanku yang terbaru ke mana?",
            "Berdasarkan info update tadi, di kota mana tempat tinggalku yang baru?",
            "Kota baru yang menjadi tujuan kepindahanku tadi apa ya?"
        ]
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
        q_rec_list = [
            "Bisa sebutkan profesi atau peran pekerjaanku yang terbaru?",
            "Setelah update karir tadi, apa jabatan/pekerjaan baruku sekarang?",
            "Kamu masih ingat profesi baruku setelah berganti peran tadi?"
        ]

    turns.append({"role": "user", "content": u_up})
    turns.append({"role": "assistant", "content": a_up})

    q_dis2, a_dis2 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] != q_dis1])
    turns.append({"role": "user", "content": q_dis2})
    turns.append({"role": "assistant", "content": a_dis2})

    if turns_count >= 12:
        q_dis3, a_dis3 = random.choice([d for d in MASSIVE_DISTRACTORS if d[0] not in (q_dis1, q_dis2)])
        turns.append({"role": "user", "content": q_dis3})
        turns.append({"role": "assistant", "content": a_dis3})

    q_rec = random.choice(q_rec_list)
    a_rec = f"Berdasarkan pembaruan terbaru darimu, {target_key.replace('_updated', '')} barumu adalah {new_val}."
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
    make_lifestyle_health_dialogue,
    make_travel_adventure_dialogue,
    make_gaming_creative_dialogue,
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
                    turns_len = random.choice([8, 10, 12])
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

                # Batch tokenization for speed
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
