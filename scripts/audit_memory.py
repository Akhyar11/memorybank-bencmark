import os

def main():
    audit_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AUDIT.md")
    if os.path.exists(audit_file):
        with open(audit_file, "r") as f:
            print(f.read())
    else:
        print("AUDIT.md not found.")

if __name__ == "__main__":
    main()
