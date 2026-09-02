"""
generate_text_dataset.py – Fixed dataset generator.

Fixes applied:
- BUG-P1-001: Dataset ambiguity fixed – queries now uniquely identify facts
  using unique entity IDs (User_XXXXX) instead of name+city combos.
- BUG-P1-002: Entity-aware splitting – all samples for a given entity go
  into the same split (train/val/test), preventing fact leakage.
- BUG-P2-003: Saves dataset/metadata.json with seed, sizes, strategy.
"""
import os
import csv
import json
import random
import hashlib
from collections import defaultdict


def generate_dataset(seed: int = 42, output_dir: str = None):
    """
    Generates a synthetic QA dataset where every query uniquely identifies
    exactly one fact.

    Strategy:
    - Each entity gets a globally unique ID: e.g. "User_042731"
    - Queries reference the entity ID directly → only one valid answer
    - Entity-aware splitting ensures no entity appears in both train and test

    Returns:
        dict with keys: train_size, val_size, test_size, output_dir
    """
    random.seed(seed)

    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'dataset'
        )
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Vocabulary pools
    # -----------------------------------------------------------------------
    cities = [
        "Jakarta", "Surabaya", "Bandung", "Medan", "Semarang",
        "Makassar", "Palembang", "Denpasar", "Balikpapan", "Manado",
        "Yogyakarta", "Malang", "Padang", "Pekanbaru", "Banjarmasin",
        "Pontianak", "Samarinda", "Mataram", "Kupang", "Ambon",
    ]
    jobs = [
        "Dokter", "Guru", "Insinyur", "Arsitek", "Programmer",
        "Polisi", "Tentara", "Pilot", "Penulis", "Wartawan",
        "Koki", "Petani", "Nelayan", "Seniman", "Desainer",
    ]
    animals = ["Kucing", "Anjing", "Burung", "Kelinci", "Ikan",
               "Kura-kura", "Hamster", "Musang", "Ayam", "Bebek"]
    pet_names = [
        "Mochi", "Luna", "Bubu", "Oyen", "Simba",
        "Milo", "Leo", "Bella", "Max", "Charlie",
        "Rocky", "Coco", "Kiko", "Molly", "Daisy",
    ]
    cars   = ["Toyota", "Honda", "Suzuki", "Mitsubishi", "Daihatsu",
              "Nissan", "Mazda", "Ford", "Hyundai", "BMW"]
    colors = ["Merah", "Biru", "Kuning", "Hijau", "Hitam",
              "Putih", "Abu-abu", "Cokelat", "Ungu", "Oranye"]

    # -----------------------------------------------------------------------
    # Entity generation – each entity gets a unique ID
    # -----------------------------------------------------------------------
    NUM_ENTITIES = 30_000  # unique people
    entities = {}  # entity_id → attributes
    for idx in range(NUM_ENTITIES):
        eid = f"User_{idx:06d}"
        entities[eid] = {
            'city':     random.choice(cities),
            'job':      random.choice(jobs),
            'animal':   random.choice(animals),
            'pet_name': random.choice(pet_names),
            'car':      random.choice(cars),
            'color':    random.choice(colors),
        }

    # -----------------------------------------------------------------------
    # Template library – every query uniquely references the entity ID
    # -----------------------------------------------------------------------
    def make_samples(eid, attrs):
        """Generate 5 (write_fact, query, answer) tuples for one entity."""
        return [
            # City
            (
                f"{eid} tinggal di kota {attrs['city']}.",
                f"Di kota mana {eid} tinggal?",
                attrs['city'],
            ),
            # Job
            (
                f"{eid} bekerja sebagai {attrs['job']}.",
                f"Apa pekerjaan {eid}?",
                attrs['job'],
            ),
            # Pet
            (
                f"{eid} memiliki peliharaan {attrs['animal']} bernama {attrs['pet_name']}.",
                f"Siapa nama peliharaan {eid}?",
                attrs['pet_name'],
            ),
            # Car brand
            (
                f"Mobil yang dimiliki {eid} adalah {attrs['car']} berwarna {attrs['color']}.",
                f"Apa merek mobil {eid}?",
                attrs['car'],
            ),
            # Car color
            (
                f"{eid} baru membeli {attrs['car']} warna {attrs['color']}.",
                f"Apa warna mobil {eid}?",
                attrs['color'],
            ),
        ]

    # -----------------------------------------------------------------------
    # Entity-aware splitting (LOCKED GROUP → same split)
    # Shuffle entity IDs, then assign 80/10/10
    # -----------------------------------------------------------------------
    entity_ids = list(entities.keys())
    random.shuffle(entity_ids)

    n          = len(entity_ids)
    train_end  = int(n * 0.80)
    val_end    = int(n * 0.90)

    train_entities = set(entity_ids[:train_end])
    val_entities   = set(entity_ids[train_end:val_end])
    test_entities  = set(entity_ids[val_end:])

    train_data, val_data, test_data = [], [], []

    for eid, attrs in entities.items():
        samples = make_samples(eid, attrs)
        if eid in train_entities:
            train_data.extend(samples)
        elif eid in val_entities:
            val_data.extend(samples)
        else:
            test_data.extend(samples)

    # Final shuffle within each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # -----------------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------------
    def save_split(data, filename):
        path = os.path.join(output_dir, filename)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['write_fact_A', 'query_B', 'expected_output_A'])
            writer.writerows(data)
        return path

    path_train = save_split(train_data, 'train.csv')
    path_val   = save_split(val_data,   'val.csv')
    path_test  = save_split(test_data,  'test.csv')

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------
    metadata = {
        "seed":              seed,
        "generator_version": "2.0.0-entity-aware",
        "num_entities":      NUM_ENTITIES,
        "train_size":        len(train_data),
        "val_size":          len(val_data),
        "test_size":         len(test_data),
        "split_strategy":    "entity_aware_80_10_10",
        "query_uniqueness":  "guaranteed_by_entity_id",
        "leakage_check":     {
            "train_val_entity_overlap": 0,
            "train_test_entity_overlap": 0,
            "val_test_entity_overlap":  0,
        },
        "files": {
            "train": "train.csv",
            "val":   "val.csv",
            "test":  "test.csv",
        }
    }
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset generated with seed={seed}:")
    print(f"  Train : {len(train_data):,} samples  → {path_train}")
    print(f"  Val   : {len(val_data):,} samples  → {path_val}")
    print(f"  Test  : {len(test_data):,} samples  → {path_test}")
    print(f"  Meta  : {meta_path}")
    print(f"\nEntity overlap check:")
    print(f"  Train ∩ Val  = {len(train_entities & val_entities)} entities")
    print(f"  Train ∩ Test = {len(train_entities & test_entities)} entities")
    print(f"  Val   ∩ Test = {len(val_entities & test_entities)} entities")

    return metadata


if __name__ == '__main__':
    generate_dataset(seed=42)
