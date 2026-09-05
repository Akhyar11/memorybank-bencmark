"""
generate_conversation_dataset.py – Multi-Turn Diverse Conversational Memory Dataset Generator.

Generates realistic multi-turn (8-12 turns) Indonesian dialogues across diverse topics:
1. Teknologi & Programming (Python, Web, AI, DevOps, Mobile)
2. Karir & Pekerjaan (Remote work, Startup, Corporate, Freelance)
3. Hobi & Hiburan (Gaming, Musik, Fotografi, Olahraga)
4. Makanan, Minuman & Diet (Kopi, Teh, Alergi, Kuliner)
5. Gaya Hidup & Kebugaran (Gym, Lari, Pola Tidur, Yoga)
6. Travel & Wisata (Destinasi lokal/internasional, Rencana liburan)
7. Hewan Peliharaan & Keluarga (Kucing, Anjing, Pasangan, Saudara)
8. Keuangan & Usaha Sampingan (Kedai kopi, Thrift shop, Investasi)

Key Features:
- 8 to 12 turns per conversation.
- 2-3 facts injected per dialogue at various turns (Turn 0, Turn 2, Turn 4).
- Distractor turns carrying coherent on-topic discussion.
- Memory recall turns (short-horizon and long-horizon).
- Memory update/correction turns (e.g., changed city, updated tech stack).
- Structured metadata tracking (facts injected, query turns, ground-truth answers).
- Standard ChatML format support for Pure Decoder-Only Next-Token Prediction.
"""
import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Rich Vocabulary & Entity Pools (Indonesian)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Dimas", "Rian", "Adit", "Rizky", "Budi", "Bayu", "Fajar", "Gilang", "Hendra", "Ilham",
    "Joko", "Kevin", "Lukman", "Maulana", "Naufal", "Oki", "Pandu", "Reza", "Satria", "Taufik",
    "Siti", "Nadia", "Alya", "Dinda", "Fira", "Gita", "Hana", "Indah", "Kartika", "Laras",
    "Maya", "Nisa", "Putri", "Rani", "Sari", "Tiara", "Vina", "Winda", "Yulia", "Zahra"
]

CITIES = [
    "Jakarta", "Surabaya", "Bandung", "Medan", "Semarang", "Makassar", "Palembang",
    "Denpasar", "Yogyakarta", "Malang", "Balikpapan", "Manado", "Padang", "Solo",
    "Banjarmasin", "Pontianak", "Bogor", "Bekasi", "Tangerang", "Depok"
]

TECH_ROLES = [
    "Frontend Developer", "Backend Developer", "Fullstack Engineer", "Data Scientist",
    "Machine Learning Engineer", "DevOps Engineer", "Mobile Developer", "UI/UX Designer",
    "Product Manager", "QA Engineer", "Cybersecurity Analyst", "Cloud Architect"
]

PROGRAMMING_LANGS = [
    "Python", "TypeScript", "JavaScript", "Golang", "Rust", "Java", "Kotlin", "Swift", "C++", "PHP"
]

FRAMEWORKS_TOOLS = [
    "React", "Vue", "Next.js", "FastAPI", "Django", "Node.js", "Flutter", "PyTorch", "Docker", "Kubernetes"
]

HOBBIES = [
    "bermain futsal", "bersepeda santai", "jogging pagi", "bermain game RPG",
    "membaca novel fiksi", "fotografi jalanan", "bermain gitar akustik",
    "belajar memasak kue", "merawat tanaman hias", "pergi ke gym"
]

GAMES = [
    "Valorant", "Mobile Legends", "Genshin Impact", "Dota 2", "Minecraft",
    "FIFA", "Elden Ring", "PUBG Mobile", "Apex Legends", "The Witcher 3"
]

FOODS = [
    "Nasi Goreng spesial", "Sate Ayam Madura", "Rendang Sapi", "Mie Ayam Bakso",
    "Gado-gado", "Soto Betawi", "Ayam Geprek sambal bawang", "Nasi Padang", "Pempek Palembang"
]

DRINKS = [
    "Kopi Espresso", "Kopi Susu Gula Aren", "Teh Hijau Matcha", "Americano dingin",
    "Jus Alpukat", "Teh Earl Grey", "Caffè Latte", "Air Kelapa muda"
]

ALLERGIES_DIETS = [
    "alergi makanan laut (seafood)", "alergi kacang-kacangan", "tidak bisa makan makanan pedas",
    "intoleransi laktosa (susu sapi)", "menjalani diet vegetarian", "alergi telur ayam"
]

PETS = [
    ("Kucing Persia", "Mochi"), ("Kucing Domestik (Oyen)", "Simba"), ("Anjing Golden Retriever", "Milo"),
    ("Kucing British Shorthair", "Luna"), ("Hamster Roborovski", "Kiko"), ("Kelinci Rex", "Bubu"),
    ("Burung Lovebird", "Chirpy"), ("Ikan Cupang hias", "Bluey")
]

TRAVEL_DESTINATIONS = [
    "Gunung Bromo", "Labuan Bajo", "Ubud Bali", "Danau Toba", "Kepulauan Raja Ampat",
    "Kawah Ijen", "Yogyakarta (Malioboro)", "Pantai Derawan", "Tokyo Jepang", "Seoul Korea"
]

SIDE_BUSINESSES = [
    "kedai kopi kecil-kecilan", "jasa pembuatan website freelance", "toko pakaian online (thrift shop)",
    "jual makanan beku (frozen food)", "kursus les privat coding", "studio foto mandiri"
]


# ---------------------------------------------------------------------------
# Dynamic Conversation Flow Builders (8 to 12 Turns)
# ---------------------------------------------------------------------------

def build_persona(entity_id: str) -> Dict[str, Any]:
    name = random.choice(FIRST_NAMES)
    city = random.choice(CITIES)
    alt_city = random.choice([c for c in CITIES if c != city])
    job = random.choice(TECH_ROLES)
    lang = random.choice(PROGRAMMING_LANGS)
    tool = random.choice(FRAMEWORKS_TOOLS)
    hobby = random.choice(HOBBIES)
    game = random.choice(GAMES)
    food = random.choice(FOODS)
    drink = random.choice(DRINKS)
    allergy = random.choice(ALLERGIES_DIETS)
    pet_type, pet_name = random.choice(PETS)
    travel = random.choice(TRAVEL_DESTINATIONS)
    side_biz = random.choice(SIDE_BUSINESSES)

    return {
        "entity_id": entity_id,
        "name": name,
        "city": city,
        "alt_city": alt_city,
        "job": job,
        "lang": lang,
        "tool": tool,
        "hobby": hobby,
        "game": game,
        "food": food,
        "drink": drink,
        "allergy": allergy,
        "pet_type": pet_type,
        "pet_name": pet_name,
        "travel": travel,
        "side_biz": side_biz,
    }


def generate_conversation_tech_career(p: Dict[str, Any], target_turns: int = 10) -> Dict[str, Any]:
    """Topic: Programming, Tech Stack, Career & Daily Work Life (8-12 turns)."""
    turns = []
    facts = []

    # Turn 0-1: Intro + Fact 1 (Name & Job & City)
    turns.append({"role": "user", "content": f"Halo! Kenalkan, aku {p['name']}. Aku sekarang kerja sebagai {p['job']} dan tinggal di {p['city']}."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']}! Senang berkenalan denganmu. Keren sekali, kamu seorang {p['job']} di {p['city']}. Ada yang bisa kubantu hari ini?"})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})

    # Turn 2-3: Tech stack discussion + Fact 2 (Language & Tool)
    turns.append({"role": "user", "content": f"Di kantorku sekarang kami lagi banyak pakai {p['lang']} dikombinasikan dengan {p['tool']}. Menurutmu apakah kombinasi ini cocok untuk proyek berskala besar?"})
    turns.append({"role": "assistant", "content": f"Kombinasi {p['lang']} dan {p['tool']} sangat solid untuk skala enterprise, {p['name']}. Arsitekturnya modular dan punya performa konkurensi yang sangat baik."})
    facts.append({"turn": 2, "key": "lang", "value": p["lang"]})
    facts.append({"turn": 2, "key": "tool", "value": p["tool"]})

    # Turn 4-5: Lifestyle / Hobby + Fact 3 (Hobby / Drink)
    turns.append({"role": "user", "content": f"Iya bener, kadang kalau lagi pusing debugging seharian, aku biasanya istirahat sambil minum {p['drink']} atau {p['hobby']}."})
    turns.append({"role": "assistant", "content": f"Pilihan istirahat yang mantap! Menikmati {p['drink']} sambil {p['hobby']} itu cara terbaik untuk meregangkan pikiran setelah berhadapan dengan baris kode."})
    facts.append({"turn": 4, "key": "drink", "value": p["drink"]})
    facts.append({"turn": 4, "key": "hobby", "value": p["hobby"]})

    # Turn 6-7: General Problem Solving / Chitchat Distractor
    turns.append({"role": "user", "content": "Btw, ada tips nggak supaya kode kita lebih mudah dites (unit testing) dan rapi?"})
    turns.append({"role": "assistant", "content": "Kuncinya ada di Dependency Injection dan memisahkan business logic dari framework. Gunakan prinsip SOLID dan buat fungsi-fungsi yang pure tanpa side-effect tersembunyi."})

    # Optional 12-turn expansion: Distractor turn 8-9
    if target_turns >= 10:
        turns.append({"role": "user", "content": "Sip, masuk akal banget. Kadang memang kita terlalu terburu-buru bikin fitur sampai lupa struktur testnya."})
        turns.append({"role": "assistant", "content": "Betul, investasi di automated testing di awal selalu menyelamatkan waktu refactoring di masa depan."})

    # Memory Recall Turn (Recall Job & City or Tech stack)
    recall_key = random.choice(["job", "city", "lang", "drink"])
    if recall_key == "job":
        q = f"Btw, kamu masih ingat apa profesiku sekarang?"
        a = f"Kamu bekerja sebagai {p['job']}."
        ans = p['job']
    elif recall_key == "city":
        q = f"Di kota mana aku tinggal tadi ya?"
        a = f"Kamu tinggal di kota {p['city']}."
        ans = p['city']
    elif recall_key == "lang":
        q = f"Bahasa pemrograman utama yang kupakai di kantor apa tadi?"
        a = f"Bahasa pemrograman utamamu adalah {p['lang']}."
        ans = p['lang']
    else:
        q = f"Minuman yang suka kuminum saat istirahat kerja tadi apa?"
        a = f"Minuman favoritmu adalah {p['drink']}."
        ans = p['drink']

    recall_turn_idx = len(turns)
    turns.append({"role": "user", "content": q})
    turns.append({"role": "assistant", "content": a})

    return {
        "topic": "tech_and_career",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": recall_turn_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q,
            "answer": a
        }
    }


def generate_conversation_lifestyle_health(p: Dict[str, Any], target_turns: int = 10) -> Dict[str, Any]:
    """Topic: Culinary, Dietary Preferences, Health & Pet (8-12 turns)."""
    turns = []
    facts = []

    # Turn 0-1: Intro + Fact 1 (Name & Food & City)
    turns.append({"role": "user", "content": f"Hai! Aku {p['name']} dari {p['city']}. Aku lagi cari ide tempat kulineran yang enak di kotaku."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']}! Senang bisa menyapa warga {p['city']}. Kamu biasanya paling suka makanan jenis apa?"})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})

    # Turn 2-3: Food preference + Allergy/Diet (Fact 2)
    turns.append({"role": "user", "content": f"Aku paling suka makan {p['food']}, tapi yang perlu diingat aku punya kondisi {p['allergy']}."})
    turns.append({"role": "assistant", "content": f"Catat, {p['name']}! Menikmati {p['food']} memang nikmat, dan sangat penting memperhatikan {p['allergy']} agar tetap aman saat jajan."})
    facts.append({"turn": 2, "key": "food", "value": p["food"]})
    facts.append({"turn": 2, "key": "allergy", "value": p["allergy"]})

    # Turn 4-5: Pet Injection (Fact 3)
    turns.append({"role": "user", "content": f"Di rumah aku juga pelihara {p['pet_type']} yang kuberi nama {p['pet_name']}. Dia suka nemenin aku pas makan."})
    turns.append({"role": "assistant", "content": f"Lucu sekali! Memelihara {p['pet_type']} bernama {p['pet_name']} pasti bikin suasana rumah selalu ramai dan ceria."})
    facts.append({"turn": 4, "key": "pet_type", "value": p["pet_type"]})
    facts.append({"turn": 4, "key": "pet_name", "value": p["pet_name"]})

    # Turn 6-7: Distractor on Health / Nutrition
    turns.append({"role": "user", "content": "Kira-kira pola makan yang bagus untuk menjaga energi seharian itu seperti apa ya?"})
    turns.append({"role": "assistant", "content": "Pastikan sarapan kaya protein dan serat, hindari karbohidrat sederhana berlebih di pagi hari, dan cukupi kebutuhan air putih minimal 2 liter per hari."})

    # Optional 10-12 turn extension
    if target_turns >= 10:
        turns.append({"role": "user", "content": "Oke siap, aku sering lupa minum air kalau sudah fokus beraktivitas seharian."})
        turns.append({"role": "assistant", "content": "Bisa pasang alarm pengingat minum atau selalu siapkan botol air 1 liter di meja kerja, {p['name']}."})

    # Memory Recall Turn
    recall_key = random.choice(["allergy", "pet_name", "food", "city"])
    if recall_key == "allergy":
        q = "Sebelum kamu rekomendasikan resto, kamu ingat pantangan atau kondisiku apa tadi?"
        a = f"Tentu, kamu memiliki kondisi {p['allergy']}."
        ans = p['allergy']
    elif recall_key == "pet_name":
        q = "Siapa nama hewan peliharaanku yang kuceritakan tadi?"
        a = f"Nama hewan peliharaanmu adalah {p['pet_name']}."
        ans = p['pet_name']
    elif recall_key == "food":
        q = "Makanan kesukaanku yang kusebutkan tadi apa ya?"
        a = f"Makanan kesukaanmu adalah {p['food']}."
        ans = p['food']
    else:
        q = "Aku tadi bilang berasal dari kota mana?"
        a = f"Kamu berasal dari kota {p['city']}."
        ans = p['city']

    recall_turn_idx = len(turns)
    turns.append({"role": "user", "content": q})
    turns.append({"role": "assistant", "content": a})

    return {
        "topic": "lifestyle_and_health",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": recall_turn_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q,
            "answer": a
        }
    }


def generate_conversation_travel_adventure(p: Dict[str, Any], target_turns: int = 10) -> Dict[str, Any]:
    """Topic: Travel, Vacation Plans, Side Business & Gaming (8-12 turns)."""
    turns = []
    facts = []

    # Turn 0-1: Intro + Fact 1 (Name & Travel destination)
    turns.append({"role": "user", "content": f"Halo! Namaku {p['name']}. Akhir tahun ini aku berencana mau liburan ke {p['travel']}."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']}! Wah, destinasi yang luar biasa! {p['travel']} punya pemandangan yang spektakuler. Sudah mulai persiapan?"})
    facts.append({"turn": 0, "key": "travel", "value": p["travel"]})

    # Turn 2-3: Side Business Fact (Fact 2)
    turns.append({"role": "user", "content": f"Iya nih lagi nabung, untungnya aku juga ada pemasukan tambahan dari {p['side_biz']} yang lagi kujalankan."})
    turns.append({"role": "assistant", "content": f"Hebat sekali! Menjalankan {p['side_biz']} butuh ketekunan tinggi, dan hasilnya pasti memuaskan untuk mendanai perjalananmu."})
    facts.append({"turn": 2, "key": "side_biz", "value": p["side_biz"]})

    # Turn 4-5: Entertainment / Gaming (Fact 3)
    turns.append({"role": "user", "content": f"Kalau lagi senggang di malam hari, biasanya aku main game {p['game']} bareng temen-temen buat refreshing."})
    turns.append({"role": "assistant", "content": f"Main {p['game']} memang seru banget untuk mabar dan melepas penat setelah seharian mengurus pekerjaan dan usaha."})
    facts.append({"turn": 4, "key": "game", "value": p["game"]})

    # Turn 6-7: Distractor on packing & preparation
    turns.append({"role": "user", "content": "Kira-kira barang apa saja yang wajib dibawa saat bepergian jauh supaya nggak kerepotan?"})
    turns.append({"role": "assistant", "content": "Powerbank berkapasitas besar, obat-obatan pribadi, pakaian cadangan tahan cuaca, serta salinan dokumen identitas digital di ponselmu."})

    # Optional 10-12 turn extension
    if target_turns >= 10:
        turns.append({"role": "user", "content": "Benar juga ya, obat pribadi sering terlewat kalau buru-buru packing."})
        turns.append({"role": "assistant", "content": "Sangat disarankan menyiapkan pouch khusus obat kecil di tas utama agar mudah dijangkau saat darurat."})

    # Memory Recall Turn
    recall_key = random.choice(["travel", "side_biz", "game"])
    if recall_key == "travel":
        q = "Tadi aku bilang mau liburan ke mana akhir tahun ini?"
        a = f"Kamu berencana liburan ke {p['travel']}."
        ans = p['travel']
    elif recall_key == "side_biz":
        q = "Usaha sampingan apa yang sedang kujalankan tadi?"
        a = f"Kamu menjalankan {p['side_biz']}."
        ans = p['side_biz']
    else:
        q = "Game apa yang biasa kumainkan bareng temen-temen?"
        a = f"Game yang biasa kamu mainkan adalah {p['game']}."
        ans = p['game']

    recall_turn_idx = len(turns)
    turns.append({"role": "user", "content": q})
    turns.append({"role": "assistant", "content": a})

    return {
        "topic": "travel_and_adventure",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": recall_turn_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q,
            "answer": a
        }
    }


def generate_conversation_memory_update(p: Dict[str, Any], target_turns: int = 10) -> Dict[str, Any]:
    """Topic: Fact Correction / Update scenario (e.g. city or tech role changes mid-dialogue)."""
    turns = []
    facts = []

    # Turn 0-1: Initial Fact (Old City & Job)
    turns.append({"role": "user", "content": f"Halo, namaku {p['name']}. Aku sekarang tinggal di {p['city']} dan kerja jadi {p['job']}."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']}! Senang berkenalan denganmu. Selamat bertugas sebagai {p['job']} di {p['city']}."})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})
    facts.append({"turn": 0, "key": "job", "value": p["job"]})

    # Turn 2-3: Distractor discussion
    turns.append({"role": "user", "content": f"Akhir-akhir ini lalu lintas di sini padat banget kalau jam berangkat kerja."})
    turns.append({"role": "assistant", "content": f"Kepadatan lalu lintas memang sering jadi tantangan di kota besar. Banyak pekerja beralih ke transportasi umum atau berangkat lebih awal."})

    # Turn 4-5: FACT UPDATE / CORRECTION (Pindah Kota ke alt_city)
    turns.append({"role": "user", "content": f"Oh iya, mulai bulan depan kantorku memutasikan aku, jadi aku resmi pindah domisili ke {p['alt_city']}."})
    turns.append({"role": "assistant", "content": f"Wah berita besar, {p['name']}! Semoga proses pindahan dan adaptasimu di {p['alt_city']} berjalan lancar ya."})
    facts.append({"turn": 4, "key": "city_updated", "value": p["alt_city"]})

    # Turn 6-7: Chitchat on new city
    turns.append({"role": "user", "content": f"Ada saran nggak hal apa yang perlu kupersiapkan pertama kali saat pindah ke kota baru?"})
    turns.append({"role": "assistant", "content": f"Cari tempat tinggal dekat fasilitas transportasi, kenali rute ke kantor baru, dan eksplor tempat makan serta kebutuhan sehari-hari di sekitar rumah."})

    # Optional 10-12 turn extension
    if target_turns >= 10:
        turns.append({"role": "user", "content": "Sip, akhir pekan ini aku mau mulai cari kos atau kontrakan di sana."})
        turns.append({"role": "assistant", "content": "Semoga dapat hunian yang nyaman, aman, dan dekat dengan tempat kerjamu!"})

    # Memory Update Recall: Must return the UPDATED city, NOT the old city!
    q = "Bisa ingatkan aku, kota domisili baruku yang baru kupindah ke mana?"
    a = f"Kota domisili barumu adalah {p['alt_city']}."
    ans = p["alt_city"]

    recall_turn_idx = len(turns)
    turns.append({"role": "user", "content": q})
    turns.append({"role": "assistant", "content": a})

    return {
        "topic": "memory_update_and_correction",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": recall_turn_idx,
            "target_key": "city_updated",
            "ground_truth": ans,
            "old_value": p["city"],
            "question": q,
            "answer": a
        }
    }


def generate_conversation_negative_denial(p: Dict[str, Any], target_turns: int = 10) -> Dict[str, Any]:
    """Scenario: User explicitly denies an assumption/mistake and provides the real fact."""
    turns = []
    facts = []

    # Turn 0-1: User introduces name and true job
    turns.append({"role": "user", "content": f"Halo, perkenalkan namaku {p['name']}."})
    turns.append({"role": "assistant", "content": f"Selamat datang {p['name']}! Senang berkenalan denganmu. Kamu sekarang bekerja di bidang apa?"})
    facts.append({"turn": 0, "key": "name", "value": p["name"]})

    # Turn 2-3: Mistaken presumption vs user denial
    fake_job = random.choice([j for j in TECH_ROLES if j != p["job"]])
    turns.append({"role": "user", "content": f"Bukan, aku bukan {fake_job}, tapi aku bekerja sebagai {p['job']}."})
    turns.append({"role": "assistant", "content": f"Mohon maaf atas kekeliruannya, {p['name']}! Baik, catatanku sudah kuperbarui: kamu berprofesi sebagai {p['job']}, bukan {fake_job}."})
    facts.append({"turn": 2, "key": "job", "value": p["job"]})

    # Turn 4-5: Discussion on true job
    turns.append({"role": "user", "content": f"Setiap hari aku sering menangani proyek menggunakan {p['lang']}."})
    turns.append({"role": "assistant", "content": f"Menarik sekali! Bahasa {p['lang']} memang sangat populer untuk kebutuhan {p['job']}."})
    facts.append({"turn": 4, "key": "lang", "value": p["lang"]})

    # Turn 6-7: Distractor
    turns.append({"role": "user", "content": "Kira-kira tips manajemen waktu yang efektif untuk pekerja seperti aku apa ya?"})
    turns.append({"role": "assistant", "content": "Gunakan teknik time-blocking, prioritaskan tugas berdasarkan urgensi dan dampak (Eisenhower Matrix), serta luangkan jeda istirahat singkat setiap 90 menit."})

    if target_turns >= 10:
        turns.append({"role": "user", "content": "Sip, akhir-akhir ini aku memang sering merasa burnout karena multitasking."})
        turns.append({"role": "assistant", "content": f"Fokus pada satu tugas utama dalam satu waktu terbukti jauh lebih efisien dan menjaga kesehatan mentalmu, {p['name']}."})

    # Recall Turn: User checks if assistant remembers the corrected job
    q = "Tadi pekerjaanku yang benar apa ya? Jangan sampai keliru lagi."
    a = f"Pekerjaanmu yang benar adalah {p['job']}."
    ans = p["job"]

    rec_idx = len(turns)
    turns.append({"role": "user", "content": q})
    turns.append({"role": "assistant", "content": a})

    return {
        "topic": "user_denial_and_correction",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": "job",
            "ground_truth": ans,
            "rejected_value": fake_job,
            "question": q,
            "answer": a
        }
    }


def generate_conversation_unknown_recall(p: Dict[str, Any], target_turns: int = 10) -> Dict[str, Any]:
    """Scenario: User asks about an unmentioned fact; Assistant must truthfully abstain / say don't know."""
    turns = []
    facts = []

    # Turn 0-1: User shares city and food
    turns.append({"role": "user", "content": f"Hai! Aku {p['name']} dari {p['city']}. Senang bisa ngobrol."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']} dari kota {p['city']}! Ada yang bisa kubantu hari ini?"})
    facts.append({"turn": 0, "key": "name", "value": p["name"]})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})

    # Turn 2-3: Food preference
    turns.append({"role": "user", "content": f"Hari ini cuacanya enak banget buat makan {p['food']} hangat."})
    turns.append({"role": "assistant", "content": f"Wah benar sekali, {p['food']} memang santapan yang pas dinikmati saat cuaca seperti ini."})
    facts.append({"turn": 2, "key": "food", "value": p["food"]})

    # Turn 4-5: Distractor
    turns.append({"role": "user", "content": "Ada rekomendasi kegiatan akhir pekan yang santai di rumah?"})
    turns.append({"role": "assistant", "content": "Membaca buku baru, menonton dokumenter menarik, atau mencoba resep masakan sederhana bisa jadi pilihan akhir pekan yang menyenangkan."})

    if target_turns >= 10:
        turns.append({"role": "user", "content": "Ide bagus, aku mau coba selesaikan buku yang belum sempat kubaca."})
        turns.append({"role": "assistant", "content": "Selamat membaca dan menikmati waktu santai di rumah ya!"})

    # Unknown Recall: Ask unmentioned entity
    unmentioned_topics = [
        ("warna mobil favoritku", "warna mobil favoritmu", "Boleh tahu apa warna mobil favoritmu?"),
        ("nama adik kandungku", "nama adik kandungmu", "Siapa nama adik kandungmu?"),
        ("golongan darahku", "golongan darahmu", "Jika boleh tahu, apa golongan darahmu?"),
        ("hobi olahragaku", "olahraga favoritmu", "Olahraga apa yang sering kamu lakukan?")
    ]
    topic_phrase, target_attr, followup = random.choice(unmentioned_topics)
    q = f"Kamu tahu nggak apa {topic_phrase}?"
    a = f"Kamu belum pernah menceritakan tentang {target_attr} sebelumnya. {followup}"
    ans = "UNKNOWN"

    rec_idx = len(turns)
    turns.append({"role": "user", "content": q})
    turns.append({"role": "assistant", "content": a})

    return {
        "topic": "unknown_fact_abstention",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": "unmentioned_fact",
            "ground_truth": ans,
            "question": q,
            "answer": a
        }
    }


def generate_conversation_negative_confirmation(p: Dict[str, Any], target_turns: int = 10) -> Dict[str, Any]:
    """Scenario: User tests assistant with a false/trick question; Assistant denies politely."""
    turns = []
    facts = []

    # Turn 0-1: User shares city
    turns.append({"role": "user", "content": f"Halo asisten! Aku {p['name']}, aku tinggal menetap di {p['city']}."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']}! Senang berkenalan dengan warga {p['city']}. Apa kabar hari ini?"})
    facts.append({"turn": 0, "key": "name", "value": p["name"]})
    facts.append({"turn": 0, "key": "city", "value": p["city"]})

    # Turn 2-3: User shares hobby/pet
    turns.append({"role": "user", "content": f"Aku punya peliharaan seekor {p['pet_type']} yang kuberi nama {p['pet_name']}."})
    turns.append({"role": "assistant", "content": f"Wah menggemaskan! Memelihara {p['pet_type']} bernama {p['pet_name']} pasti sangat menghibur harimu."})
    facts.append({"turn": 2, "key": "pet_type", "value": p["pet_type"]})
    facts.append({"turn": 2, "key": "pet_name", "value": p["pet_name"]})

    # Turn 4-5: Distractor
    turns.append({"role": "user", "content": "Bagaimana tips menjaga konsentrasi saat belajar hal baru?"})
    turns.append({"role": "assistant", "content": "Jauhkan gangguan gadget, gunakan metode Pomodoro (25 menit fokus, 5 menit istirahat), dan buat ringkasan dengan kata-katamu sendiri."})

    if target_turns >= 10:
        turns.append({"role": "user", "content": "Metode Pomodoro sering kupakai dan memang sangat membantu."})
        turns.append({"role": "assistant", "content": "Bagus sekali! Konsistensi ritme kerja seperti itu membantu otak tetap segar tanpa cepat lelah."})

    # False/Trick Question: User mentions wrong city
    wrong_city = random.choice([c for c in CITIES if c != p["city"]])
    q = f"Tadi kamu ingat kan kalau aku tinggal di {wrong_city}?"
    a = f"Bukan, kamu tadi menyebutkan bahwa kamu tinggal di {p['city']}, bukan di {wrong_city}."
    ans = f"Bukan {wrong_city}, tapi {p['city']}"

    rec_idx = len(turns)
    turns.append({"role": "user", "content": q})
    turns.append({"role": "assistant", "content": a})

    return {
        "topic": "negative_confirmation_denial",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": "city_confirmation",
            "ground_truth": ans,
            "wrong_value": wrong_city,
            "true_value": p["city"],
            "question": q,
            "answer": a
        }
    }


# ---------------------------------------------------------------------------
# Dataset Generator Pipeline
# ---------------------------------------------------------------------------

TOPIC_BUILDERS = [
    generate_conversation_tech_career,
    generate_conversation_lifestyle_health,
    generate_conversation_travel_adventure,
    generate_conversation_memory_update,
    generate_conversation_negative_denial,
    generate_conversation_unknown_recall,
    generate_conversation_negative_confirmation,
]


def format_chatml(turns: List[Dict[str, str]]) -> str:
    """Format dialogue into standard ChatML text format."""
    lines = []
    for turn in turns:
        lines.append(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>")
    return "\n".join(lines)


def generate_conversation_dataset(
    num_conversations: int = 1000,
    min_turns: int = 8,
    max_turns: int = 12,
    seed: int = 42,
    output_dir: str = "dataset"
) -> Dict[str, Any]:
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("   CONVERSATIONAL MEMORY DATASET GENERATOR (8-12 TURNS)")
    print("=" * 70)
    print(f"  Target Conversations : {num_conversations}")
    print(f"  Turn Range           : {min_turns} - {max_turns} turns")
    print(f"  Output Directory     : {os.path.abspath(output_dir)}")
    print("=" * 70)

    dataset = []
    for idx in range(num_conversations):
        eid = f"User_{idx:05d}"
        persona = build_persona(eid)
        target_turns = random.randint(min_turns, max_turns)
        builder = random.choice(TOPIC_BUILDERS)

        dialogue_data = builder(persona, target_turns=target_turns)
        chatml_text = format_chatml(dialogue_data["turns"])

        item = {
            "id": f"conv_{idx:06d}",
            "entity_id": eid,
            "topic": dialogue_data["topic"],
            "num_turns": len(dialogue_data["turns"]),
            "turns": dialogue_data["turns"],
            "chatml": chatml_text,
            "facts": dialogue_data["facts"],
            "target_recall": dialogue_data["target_recall"],
        }
        dataset.append(item)

    # Split dataset into Train (80%), Val (10%), Test (10%) with entity isolation
    random.shuffle(dataset)
    n_train = int(num_conversations * 0.8)
    n_val = int(num_conversations * 0.1)

    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train + n_val]
    test_data = dataset[n_train + n_val:]

    def save_jsonl(data: List[Dict[str, Any]], filename: str):
        path = os.path.join(output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
        print(f"  Saved {len(data):>5d} conversations -> {filename} ({os.path.getsize(path)/1024:.1f} KB)")
        return path

    save_jsonl(train_data, "conversations_train.jsonl")
    save_jsonl(val_data, "conversations_val.jsonl")
    save_jsonl(test_data, "conversations_test.jsonl")

    # Save summary metadata
    meta = {
        "generator": "generate_conversation_dataset.py",
        "total_conversations": num_conversations,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "min_turns": min_turns,
        "max_turns": max_turns,
        "topics": [
            "tech_and_career",
            "lifestyle_and_health",
            "travel_and_adventure",
            "memory_update_and_correction",
            "user_denial_and_correction",
            "unknown_fact_abstention",
            "negative_confirmation_denial"
        ]
    }
    meta_path = os.path.join(output_dir, "conversations_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n[SUCCESS] Conversational memory dataset successfully generated!")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Multi-Turn Conversational Memory Dataset")
    parser.add_argument("--num_conversations", type=int, default=1000, help="Number of conversations to generate")
    parser.add_argument("--min_turns", type=int, default=8, help="Minimum turns per conversation")
    parser.add_argument("--max_turns", type=int, default=12, help="Maximum turns per conversation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="dataset", help="Output directory")
    args = parser.parse_args()

    generate_conversation_dataset(
        num_conversations=args.num_conversations,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        seed=args.seed,
        output_dir=args.output_dir
    )
