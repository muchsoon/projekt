import subprocess

def auto_commit_push(commit_message="Automatyczna aktualizacja kodu"):
    try:
        # Dodaj wszystkie zmiany
        subprocess.run(["git", "add", "."], check=True)
        # Utwórz commit z podaną wiadomością
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        # Wypchnij zmiany do zdalnego repozytorium
        subprocess.run(["git", "push"], check=True)
        print("Repozytorium zostało zaktualizowane.")
    except subprocess.CalledProcessError as e:
        print("Błąd podczas aktualizacji repozytorium:", e)

if __name__ == '__main__':
    auto_commit_push()
