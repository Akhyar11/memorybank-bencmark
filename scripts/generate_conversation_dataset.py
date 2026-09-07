"""
generate_conversation_dataset.py – Multi-Turn Diverse Conversational Memory Dataset Generator.
Augmented with Real-world Indonesian Datasets (Evol-Instruct & ShareGPT)
and Natural Non-Templated Openings (Greetings, Casual Inquiries, Natural Chitchat).

Key Features:
- 8 to 12 turns per conversation.
- Dynamic natural openings: Sapaan santai (greetings), direct questions, natural curiosity.
- NO repetitive "Hai aku Joko" templates in Turn 0.
- Dynamic fact positions: facts appear naturally at Turn 0 or Turn 2 depending on the opening style.
- External distractors seamlessly sampled from Evol-Instruct and ShareGPT.
- Memory recall turns (short-horizon and long-horizon).
- Memory update/correction turns.
- Full metadata tracking (facts injected, query turns, ground-truth answers).
- Standard ChatML format support.
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
# Rich Vocabulary & Entity Pools (Indonesian)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Akhyar", "Dimas", "Rian", "Adit", "Rizky", "Budi", "Bayu", "Fajar", "Gilang", "Hendra", "Ilham",
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


# ---------------------------------------------------------------------------
# External Dataset Distractor Pool (Evol-Instruct & ShareGPT)
# ---------------------------------------------------------------------------

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
    """
    Loads and manages clean, natural Indonesian conversation snippets
    from Evol-Instruct and ShareGPT to use as realistic dynamic distractors.
    Supports Kaggle environment paths and local environments seamlessly.
    """
    def __init__(
        self,
        evol_path: Optional[str] = "/kaggle/input/datasets/akhyarsafrudin/memorybank-benchmark/evol-instruct-indonesian.json",
        sharegpt_path: Optional[str] = "/kaggle/input/datasets/akhyarsafrudin/memorybank-benchmark/sharegpt-indonesian.json",
        max_user_chars: int = 250,
        max_ai_chars: int = 450,
        external_ratio: float = 0.65,
    ):
        self.external_ratio = external_ratio
        self.evol_pairs: List[Tuple[str, str]] = []
        self.sharegpt_pairs: List[Tuple[str, str]] = []
        self.all_pairs: List[Tuple[str, str]] = []

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
                    for item in edata:
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

        self.all_pairs = self.sharegpt_pairs + self.evol_pairs
        print(f"  ✓ Total External Distractor Pool: {len(self.all_pairs)} natural turns available.")

    def has_data(self) -> bool:
        return len(self.all_pairs) > 0

    def sample_distractor(self) -> Tuple[str, str]:
        if not self.all_pairs:
            return (
                "Bagaimana tips menjaga konsentrasi saat belajar hal baru?",
                "Gunakan metode Pomodoro (25 menit fokus, 5 menit istirahat) dan jauhkan distraksi gadget."
            )
        return random.choice(self.all_pairs)

    def sample_evol(self) -> Tuple[str, str]:
        if self.evol_pairs:
            return random.choice(self.evol_pairs)
        return self.sample_distractor()

    def sample_sharegpt(self) -> Tuple[str, str]:
        if self.sharegpt_pairs:
            return random.choice(self.sharegpt_pairs)
        return self.sample_distractor()


# ---------------------------------------------------------------------------
# Persona & Conversation Flow Builders
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


def inject_distractor_turn(turns: List[Dict[str, str]], pool: Optional[ExternalDistractorPool], fallback_u: str, fallback_a: str):
    """Helper to inject a dynamic external distractor or fallback template."""
    if pool and pool.has_data() and random.random() < pool.external_ratio:
        u_text, a_text = pool.sample_distractor()
        turns.append({"role": "user", "content": u_text})
        turns.append({"role": "assistant", "content": a_text})
    else:
        turns.append({"role": "user", "content": fallback_u})
        turns.append({"role": "assistant", "content": fallback_a})


def generate_conversation_tech_career(p: Dict[str, Any], target_turns: int = 10, pool: Optional[ExternalDistractorPool] = None) -> Dict[str, Any]:
    """Topic: Programming, Tech Stack, Career & Daily Work Life."""
    turns = []
    facts = []

    opening_mode = random.choice(["greeting_first", "question_first", "casual_intro"])

    if opening_mode == "greeting_first":
        g_u, g_a = random.choice(CASUAL_GREETINGS)
        turns.append({"role": "user", "content": g_u})
        turns.append({"role": "assistant", "content": g_a})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Btw kenalkan, aku {p['name']}. Aku bekerja sebagai {p['job']} dan berdomisili di {p['city']}."})
        turns.append({"role": "assistant", "content": f"Senang berkenalan denganmu, {p['name']}! Seorang {p['job']} di {p['city']}. Ada topik seru yang ingin kamu bahas hari ini?"})
        facts.append({"turn": f_turn, "key": "job", "value": p["job"]})
        facts.append({"turn": f_turn, "key": "city", "value": p["city"]})
    elif opening_mode == "question_first":
        turns.append({"role": "user", "content": "Halo! Mau tanya dong tentang arsitektur sistem untuk aplikasi berskala besar."})
        turns.append({"role": "assistant", "content": "Halo! Tentu saja, pemilihan arsitektur sangat bergantung pada bahasa dan stack yang kamu gunakan. Stack apa yang biasa kamu pakai?"})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Aku {p['name']} dari {p['city']}. Di kantorku sekarang aku kerja sebagai {p['job']} dan sehari-hari banyak pakai {p['lang']} dengan {p['tool']}."})
        turns.append({"role": "assistant", "content": f"Salam kenal {p['name']}! Kombinasi {p['lang']} dan {p['tool']} sangat solid untuk skala enterprise di bidang {p['job']}."})
        facts.append({"turn": f_turn, "key": "job", "value": p["job"]})
        facts.append({"turn": f_turn, "key": "city", "value": p["city"]})
        facts.append({"turn": f_turn, "key": "lang", "value": p["lang"]})
        facts.append({"turn": f_turn, "key": "tool", "value": p["tool"]})
    else:
        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Hai, salam kenal! Panggil saja aku {p['name']}, aku berkarier sebagai {p['job']} di kota {p['city']}."})
        turns.append({"role": "assistant", "content": f"Halo {p['name']}! Salam kenal kembali. Senang bisa terhubung dengan seorang {p['job']} di {p['city']}. Ada yang bisa kubantu?"})
        facts.append({"turn": f_turn, "key": "job", "value": p["job"]})
        facts.append({"turn": f_turn, "key": "city", "value": p["city"]})

    # Middle Turn: Drink / Hobby Fact
    f2_turn = len(turns)
    turns.append({"role": "user", "content": f"Kalau lagi pusing berhadapan dengan bug seharian, biasanya aku rehat sambil minum {p['drink']} atau {p['hobby']}."})
    turns.append({"role": "assistant", "content": f"Pilihan istirahat yang mantap! Menikmati {p['drink']} sambil {p['hobby']} itu cara ampuh menyegarkan pikiran kembali."})
    facts.append({"turn": f2_turn, "key": "drink", "value": p["drink"]})
    facts.append({"turn": f2_turn, "key": "hobby", "value": p["hobby"]})

    # Distractor Turns
    inject_distractor_turn(
        turns, pool,
        fallback_u="Btw, ada tips nggak supaya kode kita lebih mudah dites (unit testing) dan rapi?",
        fallback_a="Kuncinya ada di Dependency Injection dan memisahkan business logic dari framework. Gunakan prinsip SOLID dan buat fungsi-fungsi yang modular."
    )

    if target_turns >= 10:
        inject_distractor_turn(
            turns, pool,
            fallback_u="Sip, masuk akal banget. Kadang kita terlalu terburu-buru sampai lupa struktur testnya.",
            fallback_a="Betul, investasi di automated testing di awal selalu menyelamatkan waktu refactoring di masa depan."
        )

    # Memory Recall Turn
    recall_key = random.choice(["job", "city", "drink", "hobby"])
    if recall_key == "job":
        q = "Btw, kamu masih ingat apa profesiku sekarang?"
        a = f"Kamu bekerja sebagai {p['job']}."
        ans = p['job']
    elif recall_key == "city":
        q = "Di kota mana aku tinggal tadi ya?"
        a = f"Kamu tinggal di kota {p['city']}."
        ans = p['city']
    elif recall_key == "drink":
        q = "Minuman yang suka kuminum saat istirahat tadi apa?"
        a = f"Minuman favoritmu adalah {p['drink']}."
        ans = p['drink']
    else:
        q = "Kegiatan apa yang biasa kulakukan saat rehat tadi?"
        a = f"Kamu biasa {p['hobby']} saat rehat."
        ans = p['hobby']

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


def generate_conversation_lifestyle_health(p: Dict[str, Any], target_turns: int = 10, pool: Optional[ExternalDistractorPool] = None) -> Dict[str, Any]:
    """Topic: Culinary, Dietary Preferences, Health & Pet."""
    turns = []
    facts = []

    opening_mode = random.choice(["greeting_first", "food_first", "casual_intro"])

    if opening_mode == "greeting_first":
        g_u, g_a = random.choice(CASUAL_GREETINGS)
        turns.append({"role": "user", "content": g_u})
        turns.append({"role": "assistant", "content": g_a})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Aku {p['name']} dari {p['city']}. Lagi cari ide kulineran enak nih di kotaku."})
        turns.append({"role": "assistant", "content": f"Halo {p['name']} dari {p['city']}! Siap, kamu biasanya paling gemar menyantap makanan jenis apa?"})
        facts.append({"turn": f_turn, "key": "city", "value": p["city"]})
    elif opening_mode == "food_first":
        turns.append({"role": "user", "content": "Halo! Mau tanya dong rekomendasi makanan lezat yang cocok disantap saat cuaca mendung begini."})
        turns.append({"role": "assistant", "content": "Halo! Sajian hangat dan berkuah selalu jadi pilihan favorit. Kamu lebih suka olahan daging atau yang gurih pedas?"})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Aku {p['name']} dari {p['city']}. Biasanya paling doyan makan {p['food']}, tapi aku punya kondisi {p['allergy']}."})
        turns.append({"role": "assistant", "content": f"Salam kenal {p['name']}! {p['food']} memang nikmat luar biasa, dan sangat penting mewaspadai {p['allergy']} agar kesehatanmu tetap terjaga."})
        facts.append({"turn": f_turn, "key": "city", "value": p["city"]})
        facts.append({"turn": f_turn, "key": "food", "value": p["food"]})
        facts.append({"turn": f_turn, "key": "allergy", "value": p["allergy"]})
    else:
        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Permisi, salam kenal! Aku {p['name']} warga {p['city']}. Senang bisa menyapa di sini."})
        turns.append({"role": "assistant", "content": f"Halo {p['name']}! Salam kenal warga {p['city']}. Ada topik menarik apa yang ingin kamu bahas hari ini?"})
        facts.append({"turn": f_turn, "key": "city", "value": p["city"]})

    # Middle Turn: Food + Allergy (if not injected) or Pet
    if not any(f["key"] == "food" for f in facts):
        f2_turn = len(turns)
        turns.append({"role": "user", "content": f"Aku paling doyan makan {p['food']}, tapi yang perlu kuperhatikan aku punya kondisi {p['allergy']}."})
        turns.append({"role": "assistant", "content": f"Catat, {p['name']}! Menikmati {p['food']} memang mantap, dan sangat penting memperhatikan {p['allergy']} agar selalu aman."})
        facts.append({"turn": f2_turn, "key": "food", "value": p["food"]})
        facts.append({"turn": f2_turn, "key": "allergy", "value": p["allergy"]})

    # Pet fact
    f3_turn = len(turns)
    turns.append({"role": "user", "content": f"Di rumah aku juga pelihara seekor {p['pet_type']} yang kuberi nama {p['pet_name']}. Dia suka banget nemenin pas makan."})
    turns.append({"role": "assistant", "content": f"Lucu sekali! Memelihara {p['pet_type']} bernama {p['pet_name']} pasti bikin suasana rumah selalu ceria."})
    facts.append({"turn": f3_turn, "key": "pet_name", "value": p["pet_name"]})

    # Distractor Turns
    inject_distractor_turn(
        turns, pool,
        fallback_u="Kira-kira pola makan yang bagus untuk menjaga energi seharian itu seperti apa ya?",
        fallback_a="Pastikan sarapan kaya protein dan serat, cukupi air putih minimal 2 liter, dan hindari makanan berlemak jenuh tinggi di siang hari."
    )

    if target_turns >= 10:
        inject_distractor_turn(
            turns, pool,
            fallback_u="Oke siap, aku sering lupa minum air kalau sudah fokus seharian.",
            fallback_a=f"Bisa pasang alarm pengingat minum atau selalu siapkan botol air di meja, {p['name']}."
        )

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


def generate_conversation_travel_adventure(p: Dict[str, Any], target_turns: int = 10, pool: Optional[ExternalDistractorPool] = None) -> Dict[str, Any]:
    """Topic: Travel, Vacation Plans, Side Business & Gaming."""
    turns = []
    facts = []

    opening_mode = random.choice(["greeting_first", "vacation_first", "casual_intro"])

    if opening_mode == "greeting_first":
        g_u, g_a = random.choice(CASUAL_GREETINGS)
        turns.append({"role": "user", "content": g_u})
        turns.append({"role": "assistant", "content": g_a})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Aku {p['name']}. Akhir tahun ini aku punya rencana seru mau liburan ke {p['travel']}."})
        turns.append({"role": "assistant", "content": f"Halo {p['name']}! Wah, destinasi impian! {p['travel']} punya pesona alam yang luar biasa. Sudah mulai persiapan?"})
        facts.append({"turn": f_turn, "key": "travel", "value": p["travel"]})
    elif opening_mode == "vacation_first":
        turns.append({"role": "user", "content": "Halo! Lagi butuh rekomendasi persiapan sebelum liburan ke destinasi alam terbuka nih."})
        turns.append({"role": "assistant", "content": "Halo! Menyenangkan sekali merencanakan liburan alam. Rencananya mau menjelajahi daerah mana?"})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Kenalkan aku {p['name']}. Rencanaku mau bepergian ke {p['travel']}, dananya dari tabungan {p['side_biz']} yang lagi kujalankan."})
        turns.append({"role": "assistant", "content": f"Hebat sekali, {p['name']}! Membiayai liburan ke {p['travel']} dari hasil usaha {p['side_biz']} pasti memberikan kepuasan tersendiri."})
        facts.append({"turn": f_turn, "key": "travel", "value": p["travel"]})
        facts.append({"turn": f_turn, "key": "side_biz", "value": p["side_biz"]})
    else:
        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Hai, selamat sore! Namaku {p['name']}, aku lagi semangat nabung buat liburan ke {p['travel']}."})
        turns.append({"role": "assistant", "content": f"Selamat sore {p['name']}! Semangat menabung, {p['travel']} pasti akan jadi pengalaman liburan yang tak terlupakan."})
        facts.append({"turn": f_turn, "key": "travel", "value": p["travel"]})

    # Side Biz & Gaming
    if not any(f["key"] == "side_biz" for f in facts):
        f2_turn = len(turns)
        turns.append({"role": "user", "content": f"Untungnya aku ada pemasukan sampingan dari usaha {p['side_biz']} yang lumayan membantu pendanaan liburanku."})
        turns.append({"role": "assistant", "content": f"Keren sekali! Menjalankan {p['side_biz']} butuh ketekunan tinggi, dan hasilnya sangat bernilai."})
        facts.append({"turn": f2_turn, "key": "side_biz", "value": p["side_biz"]})

    f3_turn = len(turns)
    turns.append({"role": "user", "content": f"Kalau malam hari senggang, biasanya aku mabar game {p['game']} bareng temen-temen buat refreshing."})
    turns.append({"role": "assistant", "content": f"Main {p['game']} memang seru banget untuk melepas penat setelah seharian beraktivitas dan mengurus usaha."})
    facts.append({"turn": f3_turn, "key": "game", "value": p["game"]})

    # Distractor
    inject_distractor_turn(
        turns, pool,
        fallback_u="Kira-kira barang apa saja yang wajib dibawa saat bepergian jauh supaya nggak kerepotan?",
        fallback_a="Powerbank berkapasitas besar, obat-obatan pribadi, pakaian cadangan tahan cuaca, serta dokumen identitas digital di ponsel."
    )

    if target_turns >= 10:
        inject_distractor_turn(
            turns, pool,
            fallback_u="Benar juga ya, obat pribadi sering terlewat kalau buru-buru packing.",
            fallback_a="Sangat disarankan menyiapkan pouch khusus obat kecil di tas utama agar mudah dijangkau saat darurat."
        )

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


def generate_conversation_memory_update(p: Dict[str, Any], target_turns: int = 10, pool: Optional[ExternalDistractorPool] = None) -> Dict[str, Any]:
    """Topic: Fact Correction / Update scenario (e.g. city changes mid-dialogue)."""
    turns = []
    facts = []

    opening_mode = random.choice(["greeting_first", "casual_intro"])

    if opening_mode == "greeting_first":
        g_u, g_a = random.choice(CASUAL_GREETINGS)
        turns.append({"role": "user", "content": g_u})
        turns.append({"role": "assistant", "content": g_a})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Aku {p['name']}. Saat ini aku menetap di {p['city']} dan sehari-hari bekerja sebagai {p['job']}."})
        turns.append({"role": "assistant", "content": f"Halo {p['name']}! Senang berkenalan denganmu. Selamat bertugas sebagai {p['job']} di kota {p['city']}."})
        facts.append({"turn": f_turn, "key": "city", "value": p["city"]})
        facts.append({"turn": f_turn, "key": "job", "value": p["job"]})
    else:
        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Halo salam kenal! Aku {p['name']}, aku tinggal di {p['city']} dan kerja jadi {p['job']}."})
        turns.append({"role": "assistant", "content": f"Halo {p['name']}! Senang berkenalan denganmu. Sukses selalu untuk kariermu di {p['city']}."})
        facts.append({"turn": f_turn, "key": "city", "value": p["city"]})
        facts.append({"turn": f_turn, "key": "job", "value": p["job"]})

    # Distractor discussion
    inject_distractor_turn(
        turns, pool,
        fallback_u="Akhir-akhir ini lalu lintas di kotaku padat banget kalau jam berangkat kerja.",
        fallback_a="Kepadatan lalu lintas memang sering jadi tantangan di kota besar. Banyak pekerja beralih ke transportasi umum atau berangkat lebih pagi."
    )

    # FACT UPDATE / CORRECTION (Pindah Kota ke alt_city)
    f_up_turn = len(turns)
    turns.append({"role": "user", "content": f"Oh iya, mulai bulan depan kantorku memutasikan tugasku, jadi aku resmi pindah domisili ke {p['alt_city']}."})
    turns.append({"role": "assistant", "content": f"Wah kabar penting, {p['name']}! Semoga proses kepindahan dan adaptasimu di {p['alt_city']} berjalan lancar ya."})
    facts.append({"turn": f_up_turn, "key": "city_updated", "value": p["alt_city"]})

    # Chitchat on new city
    inject_distractor_turn(
        turns, pool,
        fallback_u="Ada saran nggak hal apa yang perlu kupersiapkan pertama kali saat pindah ke kota baru?",
        fallback_a="Cari tempat tinggal dekat fasilitas transportasi, kenali rute ke kantor baru, dan eksplor tempat makan serta kebutuhan sehari-hari."
    )

    if target_turns >= 10:
        inject_distractor_turn(
            turns, pool,
            fallback_u="Sip, akhir pekan ini aku mau mulai cari tempat tinggal di sana.",
            fallback_a="Semoga dapat hunian yang nyaman, aman, dan dekat dengan lokasi kerjamu!"
        )

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


def generate_conversation_negative_denial(p: Dict[str, Any], target_turns: int = 10, pool: Optional[ExternalDistractorPool] = None) -> Dict[str, Any]:
    """Scenario: User denies an assumption and corrects fact."""
    turns = []
    facts = []

    g_u, g_a = random.choice(CASUAL_GREETINGS)
    turns.append({"role": "user", "content": g_u})
    turns.append({"role": "assistant", "content": g_a})

    f_turn = len(turns)
    fake_job = random.choice([j for j in TECH_ROLES if j != p["job"]])
    turns.append({"role": "user", "content": f"Kenalkan namaku {p['name']}. Kemarin ada yang mengira aku {fake_job}, padahal pekerjaanku yang sebenarnya adalah {p['job']}."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']}! Senang berkenalan denganmu. Sudah kucatat dengan jelas: kamu berprofesi sebagai {p['job']}, bukan {fake_job}."})
    facts.append({"turn": f_turn, "key": "job", "value": p["job"]})

    f2_turn = len(turns)
    turns.append({"role": "user", "content": f"Setiap hari aku banyak menangani sistem menggunakan {p['lang']}."})
    turns.append({"role": "assistant", "content": f"Bahasa {p['lang']} memang sangat andal dan fleksibel untuk kebutuhan seorang {p['job']}."})
    facts.append({"turn": f2_turn, "key": "lang", "value": p["lang"]})

    inject_distractor_turn(
        turns, pool,
        fallback_u="Kira-kira tips manajemen waktu yang efektif untuk pekerja sepertiku apa ya?",
        fallback_a="Gunakan teknik time-blocking, prioritaskan tugas penting, dan luangkan jeda istirahat teratur agar tidak cepat lelah."
    )

    if target_turns >= 10:
        inject_distractor_turn(
            turns, pool,
            fallback_u="Sip, belakangan ini aku memang perlu lebih disiplin mengatur waktu kerja.",
            fallback_a=f"Langkah bagus, konsistensi ritme kerja akan sangat membantu produktivitasmu, {p['name']}."
        )

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


def generate_conversation_unknown_recall(p: Dict[str, Any], target_turns: int = 10, pool: Optional[ExternalDistractorPool] = None) -> Dict[str, Any]:
    """Scenario: User asks about an unmentioned fact; Assistant must truthfully abstain."""
    turns = []
    facts = []

    g_u, g_a = random.choice(CASUAL_GREETINGS)
    turns.append({"role": "user", "content": g_u})
    turns.append({"role": "assistant", "content": g_a})

    f_turn = len(turns)
    turns.append({"role": "user", "content": f"Btw namaku {p['name']} dari kota {p['city']}. Hari ini cuacanya enak banget buat makan {p['food']} hangat."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']} warga {p['city']}! Benar sekali, menyantap {p['food']} memang pas dinikmati di cuaca seperti ini."})
    facts.append({"turn": f_turn, "key": "city", "value": p["city"]})
    facts.append({"turn": f_turn, "key": "food", "value": p["food"]})

    inject_distractor_turn(
        turns, pool,
        fallback_u="Ada rekomendasi kegiatan akhir pekan yang santai di rumah?",
        fallback_a="Membaca buku baru, menonton film dokumenter, atau mencoba resep masakan sederhana bisa jadi pilihan akhir pekan yang menyenangkan."
    )

    if target_turns >= 10:
        inject_distractor_turn(
            turns, pool,
            fallback_u="Ide bagus, aku mau coba selesaikan buku yang belum sempat kubaca.",
            fallback_a="Selamat membaca dan menikmati waktu santai di rumah ya!"
        )

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


def generate_conversation_negative_confirmation(p: Dict[str, Any], target_turns: int = 10, pool: Optional[ExternalDistractorPool] = None) -> Dict[str, Any]:
    """Scenario: User tests assistant with a false/trick question; Assistant denies politely."""
    turns = []
    facts = []

    g_u, g_a = random.choice(CASUAL_GREETINGS)
    turns.append({"role": "user", "content": g_u})
    turns.append({"role": "assistant", "content": g_a})

    f_turn = len(turns)
    turns.append({"role": "user", "content": f"Salam kenal, aku {p['name']}. Aku menetap di {p['city']} dan memelihara seekor {p['pet_type']} bernama {p['pet_name']}."})
    turns.append({"role": "assistant", "content": f"Halo {p['name']} warga {p['city']}! Memelihara {p['pet_type']} bernama {p['pet_name']} pasti sangat menghibur harimu."})
    facts.append({"turn": f_turn, "key": "city", "value": p["city"]})
    facts.append({"turn": f_turn, "key": "pet_name", "value": p["pet_name"]})

    inject_distractor_turn(
        turns, pool,
        fallback_u="Bagaimana tips menjaga konsentrasi saat belajar hal baru?",
        fallback_a="Jauhkan gangguan gadget, gunakan metode Pomodoro (25 menit fokus, 5 menit istirahat), dan buat ringkasan dengan kata-katamu sendiri."
    )

    if target_turns >= 10:
        inject_distractor_turn(
            turns, pool,
            fallback_u="Metode Pomodoro sering kupakai dan memang sangat membantu.",
            fallback_a="Bagus sekali! Ritme kerja teratur membantu otak tetap segar tanpa cepat lelah."
        )

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


def generate_conversation_evol_deep_instruct(p: Dict[str, Any], target_turns: int = 10, pool: Optional[ExternalDistractorPool] = None) -> Dict[str, Any]:
    """
    Topic: Evol-Instruct Technical & Reasoning Q&A + Episodic Persona Memory.
    Starts with a real technical question or natural greeting, facts emerge organically.
    """
    turns = []
    facts = []

    opening_mode = random.choice(["instruct_first", "greeting_first"])

    if opening_mode == "instruct_first":
        u_evol, a_evol = pool.sample_evol() if pool and pool.has_data() else (
            "Bagaimana cara mengoptimalkan query database SQL yang lambat?",
            "Gunakan indeks pada kolom WHERE/JOIN, hindari SELECT *, dan gunakan EXPLAIN ANALYZE untuk membaca execution plan."
        )
        turns.append({"role": "user", "content": u_evol})
        turns.append({"role": "assistant", "content": a_evol})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Penjelasanmu sangat jelas dan membantu! Btw kenalkan namaku {p['name']}, aku kerja sebagai {p['job']} di {p['city']}."})
        turns.append({"role": "assistant", "content": f"Senang sekali penjelasannya bermanfaat untukmu, {p['name']}! Salam sukses untuk kariermu sebagai {p['job']} di {p['city']}."})
        facts.append({"turn": f_turn, "key": "job", "value": p["job"]})
        facts.append({"turn": f_turn, "key": "city", "value": p["city"]})
    else:
        g_u, g_a = random.choice(CASUAL_GREETINGS)
        turns.append({"role": "user", "content": g_u})
        turns.append({"role": "assistant", "content": g_a})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Aku {p['name']} dari {p['city']}, sehari-hari beraktivitas sebagai {p['job']}. Mau minta panduan teknis dong."})
        turns.append({"role": "assistant", "content": f"Halo {p['name']}! Tentu saja, silakan sampaikan apa yang ingin kamu tanyakan."})
        facts.append({"turn": f_turn, "key": "job", "value": p["job"]})
        facts.append({"turn": f_turn, "key": "city", "value": p["city"]})

        u_evol, a_evol = pool.sample_evol() if pool and pool.has_data() else (
            "Bagaimana cara mengoptimalkan query database SQL yang lambat?",
            "Gunakan indeks pada kolom WHERE/JOIN, hindari SELECT *, dan gunakan EXPLAIN ANALYZE untuk membaca execution plan."
        )
        turns.append({"role": "user", "content": u_evol})
        turns.append({"role": "assistant", "content": a_evol})

    # Middle fact: drink
    f2_turn = len(turns)
    turns.append({"role": "user", "content": f"Sambil mencerna solusi tadi, aku lagi santai minum {p['drink']} nih."})
    turns.append({"role": "assistant", "content": f"Nikmat sekali! Menikmati {p['drink']} memang teman terbaik saat membaca dan belajar hal baru."})
    facts.append({"turn": f2_turn, "key": "drink", "value": p["drink"]})

    # Second Evol / ShareGPT Task
    inject_distractor_turn(
        turns, pool,
        fallback_u="Bisa berikan contoh kalimat penutup email profesional?",
        fallback_a="Tentu: 'Demikian informasi yang dapat saya sampaikan. Atas perhatian dan kerja sama Bapak/Ibu, saya ucapkan terima kasih.'"
    )

    if target_turns >= 10:
        inject_distractor_turn(
            turns, pool,
            fallback_u="Terima kasih banyak, sarannya sangat tepat guna.",
            fallback_a="Sama-sama! Selalu senang bisa membantu pekerjaanmu."
        )

    # Memory Recall
    recall_key = random.choice(["job", "city", "drink"])
    if recall_key == "job":
        q = "Ngomong-ngomong di luar topik tadi, kamu masih ingat apa profesiku?"
        a = f"Kamu bekerja sebagai {p['job']}."
        ans = p['job']
    elif recall_key == "city":
        q = "Tadi di awal aku bilang tinggal di kota mana ya?"
        a = f"Kamu tinggal di kota {p['city']}."
        ans = p['city']
    else:
        q = "Minuman yang kuminum sambil santai tadi apa ya?"
        a = f"Minuman yang kamu nikmati tadi adalah {p['drink']}."
        ans = p['drink']

    rec_idx = len(turns)
    turns.append({"role": "user", "content": q})
    turns.append({"role": "assistant", "content": a})

    return {
        "topic": "evol_instruct_reasoning_recall",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
            "question": q,
            "answer": a
        }
    }


def generate_conversation_sharegpt_natural_chat(p: Dict[str, Any], target_turns: int = 10, pool: Optional[ExternalDistractorPool] = None) -> Dict[str, Any]:
    """
    Topic: ShareGPT Natural Multi-turn Dialogues + Episodic Memory.
    Opens with natural casual conversation, facts revealed organically.
    """
    turns = []
    facts = []

    opening_mode = random.choice(["sharegpt_first", "greeting_first"])

    if opening_mode == "sharegpt_first":
        u_sg, a_sg = pool.sample_sharegpt() if pool and pool.has_data() else (
            "Apa tips sederhana untuk menata meja kerja agar tidak berantakan?",
            "Simpan hanya barang yang esensial, rapikan kabel dengan cable-tie, dan sediakan tempat sampah kecil di dekat meja."
        )
        turns.append({"role": "user", "content": u_sg})
        turns.append({"role": "assistant", "content": a_sg})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Ide yang bagus! Namaku {p['name']}. Meja kerjaku sering berantakan karena suka dinaiki peliharaanku seekor {p['pet_type']} bernama {p['pet_name']}."})
        turns.append({"role": "assistant", "content": f"Wah lucu sekali, {p['name']}! Tingkah {p['pet_type']} bernama {p['pet_name']} pasti sangat menggemaskan walau bikin meja berantakan."})
        facts.append({"turn": f_turn, "key": "pet_name", "value": p["pet_name"]})
    else:
        g_u, g_a = random.choice(CASUAL_GREETINGS)
        turns.append({"role": "user", "content": g_u})
        turns.append({"role": "assistant", "content": g_a})

        f_turn = len(turns)
        turns.append({"role": "user", "content": f"Aku {p['name']}. Baru saja selesai {p['hobby']} dan sekarang lagi mau ngobrol santai."})
        turns.append({"role": "assistant", "content": f"Halo {p['name']}! Pasti segar rasanya setelah {p['hobby']}. Ada hal seru apa yang ingin kamu bahas?"})
        facts.append({"turn": f_turn, "key": "hobby", "value": p["hobby"]})

        u_sg, a_sg = pool.sample_sharegpt() if pool and pool.has_data() else (
            "Apa tips sederhana untuk menata meja kerja agar tidak berantakan?",
            "Simpan hanya barang yang esensial, rapikan kabel dengan cable-tie, dan sediakan tempat sampah kecil di dekat meja."
        )
        turns.append({"role": "user", "content": u_sg})
        turns.append({"role": "assistant", "content": a_sg})

    # Second ShareGPT distractor
    inject_distractor_turn(
        turns, pool,
        fallback_u="Bisa beri saran aktivitas yang menenangkan pikiran setelah seharian sibuk?",
        fallback_a="Mendengarkan musik akustik instrumental, berjalan santai di luar ruangan, atau menyeduh teh hangat tanpa gadget."
    )

    if target_turns >= 10:
        inject_distractor_turn(
            turns, pool,
            fallback_u="Bagus juga sarannya, nanti malam mau coba jalan santai sebentar.",
            fallback_a="Selamat beristirahat dan menikmati malam yang tenang ya!"
        )

    # Memory Recall
    recall_key = "pet_name" if any(f["key"] == "pet_name" for f in facts) else "hobby"
    if recall_key == "pet_name":
        q = "Siapa nama hewan peliharaanku yang suka main di meja kerja tadi?"
        a = f"Nama hewan peliharaanmu adalah {p['pet_name']}."
        ans = p['pet_name']
    else:
        q = "Sebelum kita ngobrol tadi, aku bilang habis selesai melakukan kegiatan apa?"
        a = f"Kamu tadi baru saja selesai {p['hobby']}."
        ans = p['hobby']

    rec_idx = len(turns)
    turns.append({"role": "user", "content": q})
    turns.append({"role": "assistant", "content": a})

    return {
        "topic": "sharegpt_natural_dialogue_recall",
        "turns": turns,
        "facts": facts,
        "target_recall": {
            "query_turn": rec_idx,
            "target_key": recall_key,
            "ground_truth": ans,
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
    generate_conversation_evol_deep_instruct,
    generate_conversation_sharegpt_natural_chat,
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
    output_dir: str = "dataset",
    evol_path: Optional[str] = "/kaggle/input/datasets/akhyarsafrudin/memorybank-benchmark/evol-instruct-indonesian.json",
    sharegpt_path: Optional[str] = "/kaggle/input/datasets/akhyarsafrudin/memorybank-benchmark/sharegpt-indonesian.json",
    external_distractor_ratio: float = 0.65,
) -> Dict[str, Any]:
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("   CONVERSATIONAL MEMORY DATASET GENERATOR (NATURAL OPENINGS + REAL DATA)")
    print("=" * 72)
    print(f"  Target Conversations      : {num_conversations}")
    print(f"  Turn Range                : {min_turns} - {max_turns} turns")
    print(f"  Output Directory          : {os.path.abspath(output_dir)}")
    print(f"  External Distractor Ratio : {external_distractor_ratio * 100:.0f}%")
    print("=" * 72)

    print("\n[STEP 1]: Loading Real External Indonesian Datasets...")
    pool = ExternalDistractorPool(
        evol_path=evol_path,
        sharegpt_path=sharegpt_path,
        external_ratio=external_distractor_ratio,
    )

    print(f"\n[STEP 2]: Generating {num_conversations} Diverse Conversations with Natural Openings...")
    dataset = []
    topic_counts = {}

    for idx in range(num_conversations):
        eid = f"User_{idx:05d}"
        persona = build_persona(eid)
        target_turns = random.randint(min_turns, max_turns)
        builder = random.choice(TOPIC_BUILDERS)

        dialogue_data = builder(persona, target_turns=target_turns, pool=pool)
        chatml_text = format_chatml(dialogue_data["turns"])

        top = dialogue_data["topic"]
        topic_counts[top] = topic_counts.get(top, 0) + 1

        item = {
            "id": f"conv_{idx:06d}",
            "entity_id": eid,
            "topic": top,
            "num_turns": len(dialogue_data["turns"]),
            "turns": dialogue_data["turns"],
            "chatml": chatml_text,
            "facts": dialogue_data["facts"],
            "target_recall": dialogue_data["target_recall"],
        }
        dataset.append(item)

    # Split dataset into Train (80%), Val (10%), Test (10%)
    random.shuffle(dataset)
    n_train = int(num_conversations * 0.8)
    n_val = int(num_conversations * 0.1)

    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train + n_val]
    test_data = dataset[n_train + n_val:]

    def save_jsonl(data: List[Dict[str, Any]], filename: str):
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  ✓ Saved {len(data):>5d} conversations -> {filename} ({os.path.getsize(path)/1024:.1f} KB)")
        return path

    print("\n[STEP 3]: Exporting Split JSONL Files...")
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
        "features": [
            "natural_openings_greetings",
            "non_templated_intro",
            "dynamic_fact_turns",
            "evol_instruct_distractors",
            "sharegpt_distractors"
        ],
        "external_datasets": {
            "evol_instruct_pool_size": len(pool.evol_pairs),
            "sharegpt_pool_size": len(pool.sharegpt_pairs),
            "total_external_turns": len(pool.all_pairs),
            "external_distractor_ratio": external_distractor_ratio
        },
        "topic_distribution": topic_counts,
        "topics": list(topic_counts.keys())
    }
    meta_path = os.path.join(output_dir, "conversations_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n--- DISTRIBUSI TOPIK DATASET ---")
    for top, cnt in topic_counts.items():
        pct = (cnt / num_conversations) * 100.0
        print(f"  {top:<35}: {cnt:>4d} dialog ({pct:5.1f}%)")
    print("=" * 72)
    print("✓ Dataset percakapan dengan pembukaan alami & augmentasi real data berhasil dibuat!")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Multi-Turn Conversational Memory Dataset Augmented with Evol-Instruct & ShareGPT")
    parser.add_argument("--num_conversations", type=int, default=1000, help="Number of conversations to generate")
    parser.add_argument("--min_turns", type=int, default=8, help="Minimum turns per conversation")
    parser.add_argument("--max_turns", type=int, default=12, help="Maximum turns per conversation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="dataset", help="Output directory")
    parser.add_argument(
        "--evol_path",
        type=str,
        default="/kaggle/input/datasets/akhyarsafrudin/memorybank-benchmark/evol-instruct-indonesian.json",
        help="Path to evol-instruct-indonesian.json (Kaggle or local)"
    )
    parser.add_argument(
        "--sharegpt_path",
        type=str,
        default="/kaggle/input/datasets/akhyarsafrudin/memorybank-benchmark/sharegpt-indonesian.json",
        help="Path to sharegpt-indonesian.json (Kaggle or local)"
    )
    parser.add_argument("--external_ratio", type=float, default=0.65, help="Ratio of distractors taken from external datasets")
    args = parser.parse_args()

    generate_conversation_dataset(
        num_conversations=args.num_conversations,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        seed=args.seed,
        output_dir=args.output_dir,
        evol_path=args.evol_path,
        sharegpt_path=args.sharegpt_path,
        external_distractor_ratio=args.external_ratio
    )
