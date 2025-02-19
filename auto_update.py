import sys
import os
import time
import subprocess

CHECK_INTERVAL = 60  # czas w sekundach między kolejnymi sprawdzeniami

def git_pull():
    """
    Wykonuje polecenie 'git pull' i zwraca True, jeśli pobrano nowe zmiany.
    """
    try:
        # Uruchomienie polecenia git pull
        result = subprocess.run(["git", "pull"], capture_output=True, text=True)
        output = result.stdout.strip()
        print(output)
        # Jeśli output nie zawiera informacji "Already up to date.",
        # to znaczy, że pobrano nowe zmiany.
        if "Already up to date." not in output:
            return True
    except Exception as e:
        print(f"Git pull failed: {e}")
    return False

def main():
    print("Automatyczne aktualizowanie kodu rozpoczęte...")
    while True:
        if git_pull():
            print("Nowe zmiany zostały pobrane. Restartowanie aplikacji...")
            # Restart skryptu – zastępujemy bieżący proces nowym
            os.execv(sys.executable, [sys.executable] + sys.argv)
        # Czekamy określony czas przed kolejnym sprawdzeniem
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main()
