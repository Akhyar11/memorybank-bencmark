import os
import csv
import random

def generate_dataset():
    dataset = []

    first_names = ["Budi", "Andi", "Joko", "Siti", "Ayu", "Dewi", "Tono", "Wati", "Eko", "Rini", "Rudi", "Nina", "Hadi", "Lia", "Agus", "Dian", "Hendra", "Maya", "Dedy", "Rina", "Bayu", "Fitri", "Aris", "Nita", "Doni", "Sari", "Rio", "Mega", "Iwan", "Putri"]
    last_names = ["Santoso", "Wijaya", "Kusuma", "Setiawan", "Hidayat", "Pratama", "Putra", "Saputra", "Wibowo", "Nugroho", "Gunawan", "Rahman", "Suryono", "Utama", "Purnama", "Siregar", "Nasution", "Simanjuntak", "Halim", "Lim"]
    cities = ["Jakarta", "Surabaya", "Bandung", "Medan", "Semarang", "Makassar", "Palembang", "Denpasar", "Balikpapan", "Manado", "Yogyakarta", "Malang", "Padang", "Pekanbaru", "Banjarmasin", "Pontianak", "Samarinda", "Mataram", "Kupang", "Ambon"]
    jobs = ["Dokter", "Guru", "Insinyur", "Arsitek", "Programmer", "Polisi", "Tentara", "Pilot", "Pramugari", "Penulis", "Wartawan", "Koki", "Petani", "Nelayan", "Seniman", "Desainer", "Aktor", "Penyanyi", "Atlet", "Pengusaha"]
    animals = ["Kucing", "Anjing", "Burung", "Kelinci", "Ikan", "Kura-kura", "Hamster", "Musang", "Ayam", "Bebek"]
    pet_names = ["Mochi", "Luna", "Bubu", "Oyen", "Simba", "Milo", "Leo", "Bella", "Max", "Charlie", "Rocky", "Coco", "Kiko", "Molly", "Daisy", "Lucy", "Bela", "Chico", "Zorro", "Nemo"]
    colors = ["Merah", "Biru", "Kuning", "Hijau", "Hitam", "Putih", "Abu-abu", "Cokelat", "Ungu", "Oranye", "Merah Muda", "Emas", "Perak"]
    cars = ["Toyota", "Honda", "Suzuki", "Mitsubishi", "Daihatsu", "Nissan", "Mazda", "Ford", "Chevrolet", "Hyundai", "Kia", "BMW", "Mercedes-Benz", "Audi", "Lexus"]

    random.seed(42)

    # Generate 25000 Personal Info
    for _ in range(25000):
        first = random.choice(first_names)
        last = random.choice(last_names)
        city = random.choice(cities)
        dataset.append((f"Nama lengkap orang ini adalah {first} {last}, dan dia berasal dari {city}.", f"Siapa nama lengkap orang yang berasal dari {city} ini?", f"{first} {last}"))
        dataset.append((f"{first} {last} lahir dan dibesarkan di kota {city}.", f"Di mana {first} {last} lahir dan dibesarkan?", f"{city}"))

    # Generate 25000 Jobs
    for _ in range(25000):
        first = random.choice(first_names)
        last = random.choice(last_names)
        job = random.choice(jobs)
        dataset.append((f"{first} {last} bekerja sebagai seorang {job} profesional.", f"Apa pekerjaan dari {first} {last}?", f"{job}"))

    # Generate 25000 Pets
    for _ in range(25000):
        first = random.choice(first_names)
        animal = random.choice(animals)
        pet = random.choice(pet_names)
        dataset.append((f"{first} memiliki peliharaan {animal} yang diberi nama {pet}.", f"Siapa nama {animal} peliharaan {first}?", f"{pet}"))

    # Generate 25000 Cars
    for _ in range(25000):
        first = random.choice(first_names)
        car = random.choice(cars)
        color = random.choice(colors)
        dataset.append((f"Kendaraan sehari-hari {first} adalah mobil {car} berwarna {color}.", f"Apa merek dan warna mobil {first}?", f"{car} berwarna {color}"))
        dataset.append((f"{first} baru saja membeli mobil {car} yang warnanya {color}.", f"Apa warna mobil {car} yang dibeli {first}?", f"{color}"))

    # Generate 25000 Passwords/Codes
    for _ in range(25000):
        code = str(random.randint(100000, 999999))
        user = f"User_{random.randint(100, 999)}"
        dataset.append((f"Kode rahasia untuk {user} adalah {code}.", f"Apa kode rahasia untuk {user}?", f"{code}"))

    # Remove duplicates
    dataset = list(set(dataset))

    # Shuffle
    random.shuffle(dataset)

    # Split dataset: 80% Train, 10% Val, 10% Test
    total = len(dataset)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train_data = dataset[:train_end]
    val_data = dataset[train_end:val_end]
    test_data = dataset[val_end:]

    # Save to CSV
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    def save_split(data, filename):
        csv_path = os.path.join(output_dir, filename)
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["write_fact_A", "query_B", "expected_output_A"])
            for row in data:
                writer.writerow(row)
        return csv_path

    path_train = save_split(train_data, "train.csv")
    path_val = save_split(val_data, "val.csv")
    path_test = save_split(test_data, "test.csv")

    print(f"Dataset berhasil dibagi dan disimpan di {output_dir}:")
    print(f" - Train : {len(train_data)} baris -> {path_train}")
    print(f" - Val   : {len(val_data)} baris -> {path_val}")
    print(f" - Test  : {len(test_data)} baris -> {path_test}")
    print(f"Total pasangan data unik: {total}")
    
    # Cetak 5 sampel pertama dari Test set
    print("\n[PREVIEW 5 DATA PERTAMA DARI TEST SET]")
    print("-" * 70)
    for i in range(min(5, len(test_data))):
        print(f"Write A (Input)   : {test_data[i][0]}")
        print(f"Query B (Tanya)   : {test_data[i][1]}")
        print(f"Expected (Jawab)  : {test_data[i][2]}")
        print("-" * 70)

if __name__ == "__main__":
    generate_dataset()
