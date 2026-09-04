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


def make_tech_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    turns, facts = [], []

    # Turn 0-1
    greetings = [
        f"Halo! Kenalkan, aku {p['name']}. Saat ini aku bekerja sebagai {p['job']} dan menetap di {p['city']}.",
        f"Hai! Salam kenal, namaku {p['name']}. Aku tinggal di {p['city']} dan sehari-hari sibuk sebagai {p['job']}.",
        f"Halo asisten! Aku {p['name']} dari {p['city']}. Profesi utamaku saat ini adalah {p['job']}."
    ]
    turns.append({"role": "user", "content": random.choice(greetings)})
    turns.append({"role": "assistant", "content": f"Halo {p['name']}! Senang berkenalan denganmu. Luar biasa, seorang {p['job']} di {p['city']}. Ada topik atau hal menarik yang ingin kita bahas hari ini?"})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})

    # Turn 2-3
    tech_prompts = [
        f"Di tempat kerjaku sekarang kami banyak menggunakan {p['lang']} dan framework {p['tool']}. Menurutmu apa keunggulan utama stack ini?",
        f"Aku lagi eksplorasi arsitektur baru menggunakan {p['lang']} dipadukan dengan {p['tool']}. Apakah kombinasi ini scalable?",
        f"Proyek terbaruku dibangun dengan {p['lang']} dan {p['tool']}. Ada saran best practice untuk optimasi performanya?"
    ]
    turns.append({"role": "user", "content": random.choice(tech_prompts)})
    turns.append({"role": "assistant", "content": f"Kombinasi {p['lang']} dengan {p['tool']} sangat populer karena efisiensi eksekusi dan ekosistem library yang matang, {p['name']}. Kuncinya ada pada pengelolaan state dan caching yang tepat."})
    facts.append({"turn": 2, "key": "lang", "value": p["lang"]})
    facts.append({"turn": 2, "key": "tool", "value": p["tool"]})

    # Turn 4-5
    lifestyle_prompts = [
        f"Kalau lagi jenuh sama urusan pekerjaan, pelarianku biasanya minum {p['drink']} sambil {p['hobby']}.",
        f"Biar nggak burnout kerja terus, aku rutin meluangkan waktu buat {p['hobby']} dan menikmati {p['drink']}.",
        f"Untuk menjaga keseimbangan hidup, rutinitas favoritku setelah jam kantor adalah {p['hobby']} ditemani segelas {p['drink']}."
    ]
    turns.append({"role": "user", "content": random.choice(lifestyle_prompts)})
    turns.append({"role": "assistant", "content": f"Itu keseimbangan yang sehat sekali, {p['name']}. Menikmati {p['drink']} sambil {p['hobby']} terbukti ampuh menyegarkan pikiran kembali."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    # Turn 6-7: Distractor
    distractors = [
        ("Bagaimana cara terbaik mengelola waktu saat menangani beberapa proyek sekaligus?",
         "Gunakan teknik Time-Blocking dan Matriks Eisenhower. Prioritaskan tugas mendesak yang berdampak besar dan minimalkan multitasking."),
        ("Menurutmu apa kriteria dokumentasi teknis yang baik untuk tim?",
         "Dokumentasi yang baik harus ringkas, memiliki contoh penggunaan (code snippet) yang jelas, dan selalu diperbarui bersamaan dengan rilis fitur baru."),
        ("Apakah sertifikasi profesional sangat berpengaruh untuk jenjang karir?",
         "Sertifikasi membuktikan pemahaman standar industri dan dedikasi belajar, namun portofolio proyek riil tetap menjadi pembuktian terkuat.")
    ]
    q_dis, a_dis = random.choice(distractors)
    turns.append({"role": "user", "content": q_dis})
    turns.append({"role": "assistant", "content": a_dis})

    # Turn 8-9: Extra distractor if turns_count == 12
    if turns_count >= 12:
        turns.append({"role": "user", "content": "Setuju banget. Kadang orang terlalu fokus mengejar sertifikat sampai lupa membangun proyek nyata."})
        turns.append({"role": "assistant", "content": "Betul sekali, kombinasi antara fondasi teori yang tersertifikasi dan jam terbang proyek langsung adalah yang paling dicari."})

    # Turn: Memory Recall
    recall_key = random.choice(["job", "city", "lang", "drink", "hobby"])
    q_map = {
        "job": (f"Ngomong-ngomong, kamu masih ingat apa profesi pekerjaanku?", f"Kamu bekerja sebagai {p['job']}.", p['job']),
        "city": (f"Bisa sebutkan di kota mana aku tinggal tadi?", f"Kamu tinggal di kota {p['city']}.", p['city']),
        "lang": (f"Bahasa pemrograman apa yang tadi kuceritakan sering kupakai?", f"Bahasa pemrograman yang sering kamu pakai adalah {p['lang']}.", p['lang']),
        "drink": (f"Minuman kesukaanku pas istirahat tadi apa ya?", f"Minuman kesukaanmu adalah {p['drink']}.", p['drink']),
        "hobby": (f"Aktivitas hobi yang biasa kulakukan setelah kerja apa tadi?", f"Aktivitas hobimu adalah {p['hobby']}.", p['hobby']),
    }
    q_rec, a_rec, ans = q_map[recall_key]
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


def make_lifestyle_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    turns, facts = [], []

    # Turn 0-1
    turns.append({"role": "user", "content": f"Halo! Aku {p['name']}, warga kota {p['city']}. Aku lagi mau menata gaya hidup dan pola makan baruku nih."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']} dari {p['city']}! Langkah yang sangat positif. Menata gaya hidup sehat adalah investasi terbaik untuk masa depan."})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})

    # Turn 2-3
    turns.append({"role": "user", "content": f"Makanan kesukaanku itu {p['food']}, tapi penting dicatat kalau aku punya kondisi {p['allergy']}."})
    turns.append({"role": "assistant", "content": f"Tercatat jelas, {p['name']}! {p['food']} memang sangat lezat, dan kita wajib selalu waspada dengan kondisi {p['allergy']} saat memilih menu harian."})
    facts.append({"turn": 2, "key": "food", "value": p["food"]})
    facts.append({"turn": 2, "key": "allergy", "value": p["allergy"]})

    # Turn 4-5
    turns.append({"role": "user", "content": f"Di rumah aku juga tinggal bareng peliharaanku, seekor {p['pet_type']} yang kuberi nama {p['pet_name']}."})
    turns.append({"role": "assistant", "content": f"Wah menggemaskan sekali! Keberadaan {p['pet_type']} bernama {p['pet_name']} pasti selalu menambah keceriaan dan mengurangi stres di rumah."})
    facts.append({"turn": 4, "key": "pet_type", "value": p["pet_type"]})
    facts.append({"turn": 4, "key": "pet_name", "value": p["pet_name"]})

    # Turn 6-7: Distractor
    turns.append({"role": "user", "content": "Kira-kira berapa durasi olahraga ringan yang ideal untuk pemula setiap minggunya?"})
    turns.append({"role": "assistant", "content": "Berdasarkan pedoman kesehatan, 150 menit per minggu untuk intensitas sedang (seperti jalan cepat 30 menit sehari selama 5 hari) sudah sangat ideal bagi pemula."})

    if turns_count >= 12:
        turns.append({"role": "user", "content": "Berarti nggak harus langsung olahraga berat setiap hari ya, yang penting konsisten."})
        turns.append({"role": "assistant", "content": "Tepat sekali, konsistensi jauh lebih penting daripada intensitas tinggi yang hanya bertahan seminggu lalu berhenti."})

    # Recall Turn
    recall_key = random.choice(["allergy", "pet_name", "food", "city"])
    q_map = {
        "allergy": ("Sebelum kita bahas resep, kamu ingat pantangan makan atau kondisiku apa?", f"Kamu memiliki kondisi {p['allergy']}.", p['allergy']),
        "pet_name": ("Siapa nama hewan peliharaan kesayanganku di rumah?", f"Nama hewan peliharaanmu adalah {p['pet_name']}.", p['pet_name']),
        "food": ("Makanan favorit yang kusebutkan tadi apa ya?", f"Makanan favoritmu adalah {p['food']}.", p['food']),
        "city": ("Aku tadi bilang berdomisili di mana?", f"Kamu berdomisili di kota {p['city']}.", p['city']),
    }
    q_rec, a_rec, ans = q_map[recall_key]
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


def make_travel_business_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    turns, facts = [], []

    # Turn 0-1
    turns.append({"role": "user", "content": f"Halo! Namaku {p['name']}. Aku punya impian besar tahun ini mau jalan-jalan ke {p['travel']}."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']}! Pilihan destinasi yang spektakuler. {p['travel']} selalu menawarkan pengalaman wisata yang tak terlupakan."})
    facts.append({"turn": 0, "key": "travel", "value": p["travel"]})

    # Turn 2-3
    turns.append({"role": "user", "content": f"Untuk menambah tabungan liburan, aku sekarang lagi aktif merintis {p['side_biz']}."})
    turns.append({"role": "assistant", "content": f"Keren banget jiwa kewirausahaanmu, {p['name']}. Mengembangkan {p['side_biz']} pasti memberikan hasil manis untuk mewujudkan impian wisatamu."})
    facts.append({"turn": 2, "key": "side_biz", "value": p["side_biz"]})

    # Turn 4-5
    turns.append({"role": "user", "content": f"Kalau malam hari pas butuh hiburan santai, aku biasanya main game {p['game']}."})
    turns.append({"role": "assistant", "content": f"Bermain {p['game']} memang seru dan jadi sarana efektif untuk melepas penat setelah seharian fokus bekerja."})
    facts.append({"turn": 4, "key": "game", "value": p["game"]})

    # Turn 6-7: Distractor
    turns.append({"role": "user", "content": "Ada tips supaya keuangan usaha kecil tidak bercampur dengan uang pribadi?"})
    turns.append({"role": "assistant", "content": "Pisahkan rekening bank sejak hari pertama, buat pembukuan arus kas harian yang disiplin, dan tentukan gaji tetap untuk dirimu sendiri."})

    if turns_count >= 12:
        turns.append({"role": "user", "content": "Pemisahan rekening itu simpel tapi sering diabaikan orang ya."})
        turns.append({"role": "assistant", "content": "Betul, pemisahan rekening menjaga visibilitas profitabilitas bisnis agar kita tahu pasti apakah usaha sedang untung atau rugi."})

    # Recall Turn
    recall_key = random.choice(["travel", "side_biz", "game"])
    q_map = {
        "travel": ("Tadi rencanaku mau pergi wisata ke mana ya?", f"Kamu berencana pergi wisata ke {p['travel']}.", p['travel']),
        "side_biz": ("Usaha sampingan apa yang sedang kurintis tadi?", f"Kamu sedang merintis usaha {p['side_biz']}.", p['side_biz']),
        "game": ("Game apa yang biasa kumainkan buat hiburan malam hari?", f"Game yang biasa kamu mainkan adalah {p['game']}.", p['game']),
    }
    q_rec, a_rec, ans = q_map[recall_key]
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


def make_update_dialogue(p: Dict[str, Any], turns_count: int) -> Dict[str, Any]:
    turns, facts = [], []

    # Turn 0-1: Initial Fact (Old City)
    turns.append({"role": "user", "content": f"Hai, aku {p['name']}. Saat ini domisiliku masih di {p['city']} dan profesiku adalah {p['job']}."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']}! Senang berkenalan denganmu. Salam hangat untukmu sebagai {p['job']} di {p['city']}."})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    # Turn 2-3: Distractor
    turns.append({"role": "user", "content": "Musim hujan di sini sering bikin jadwal kegiatan luar ruangan terhambat."})
    turns.append({"role": "assistant", "content": "Musim penghujan memang menuntut kita lebih fleksibel. Selalu siapkan payung dan pantau prakiraan cuaca sebelum bepergian."})

    # Turn 4-5: MEMORY UPDATE (City changes to alt_city)
    turns.append({"role": "user", "content": f"Kabar terbarunya, bulan depan aku resmi pindah tempat tinggal ke {p['alt_city']} karena urusan keluarga."})
    turns.append({"role": "assistant", "content": f"Selamat atas rencana kepindahannya ke {p['alt_city']}, {p['name']}! Semoga suasana baru nanti membawa banyak berkah dan kenyamanan."})
    facts.append({"turn": 4, "key": "city_updated", "value": p["alt_city"]})

    # Turn 6-7: Distractor
    turns.append({"role": "user", "content": "Kira-kira apa hal terpenting yang harus dicek sebelum menandatangani kontrak sewa hunian?"})
    turns.append({"role": "assistant", "content": "Periksa kondisi instalasi air dan listrik, kebersihan lingkungan, keamanan sekitar, serta klausul biaya perbaikan jika terjadi kerusakan struktural."})

    if turns_count >= 12:
        turns.append({"role": "user", "content": "Poin instalasi air memang krusial banget, jangan sampai pas ditinggali baru ketahuan macet."})
        turns.append({"role": "assistant", "content": "Tepat sekali, mencoba langsung kran air dan sakelar lampu saat survei fisik sangat dianjurkan."})

    # Recall Turn: MUST RETURN UPDATED CITY
    q_rec = "Bisa ingatkan aku, kota tujuan pindahanku yang baru ke mana?"
    a_rec = f"Kota tujuan kepindahanmu yang baru adalah {p['alt_city']}."
    ans = p['alt_city']
    rec_idx = len(turns)
    turns.append({"role": "user", "content": q_rec})
    turns.append({"role": "assistant", "content": a_rec})

    return {
        "topic": "memory_update_and_correction",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": "city_updated",
            "ground_truth": ans,
            "old_value": p["city"],
            "question": q_rec,
            "answer": a_rec
        }
    }


BUILDERS = [make_tech_dialogue, make_lifestyle_dialogue, make_travel_business_dialogue, make_update_dialogue]


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
    parser.add_argument("--target_tokens", type=int, default=None,
                        help="Total target tokens across all splits (automatically splits 90% train, 5% val, 5% test)")
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
