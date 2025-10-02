import pyperclip
import pyautogui
import time

# --- Configuration ---
# Set your desired Words Per Minute (WPM)
WPM = 1000

# Seconds to wait before the script starts typing.
# This gives you time to switch to the correct window.
START_DELAY = 20

# --- Script Logic ---

def auto_typer(wpm):
    """
    Gets text from the clipboard and types it at a specified WPM.
    """
    try:
        # Based on the standard that an average English word is 5 characters long.
        # Calculate Characters Per Second (CPS)
        chars_per_minute = wpm * 5
        delay_between_chars = 60 / chars_per_minute

        print("--- Auto Typer Initialized ---")
        print(f"Typing speed set to: {wpm} WPM")
        print(f"The script will start in {START_DELAY} seconds.")
        print("Please copy your text and click on the window where you want to type.")

        # Countdown timer
        for i in range(START_DELAY, 0, -1):
            print(f"{i}...", end="", flush=True)
            time.sleep(1)
        
        print("\nTyping now!")

        # 1. Get the content from the clipboard
        text_to_type = pyperclip.paste()

        # 2. Type it out with the calculated delay
        if text_to_type:
            pyautogui.write(text_to_type, interval=delay_between_chars)
            print("\n--- Typing Complete! ---")
        else:
            print("\n--- Clipboard is empty. No text to type. ---")

    except KeyboardInterrupt:
        print("\n--- Script interrupted by user. Exiting. ---")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    auto_typer(WPM)