from PIL import Image
import customtkinter as ctk
import sqlite3
import pyttsx3
from customtkinter import CTkImage

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Database setup
def get_books_in_cabinet(cabinet):
    """Fetch books from the database for the specified cabinet."""
    try:
        conn = sqlite3.connect('Data.db')
        cursor = conn.cursor()
        cursor.execute('SELECT Titles FROM BookInfo WHERE Location LIKE ?', (f"%{cabinet}%",))
        books = cursor.fetchall()
        conn.close()
        return books
    except Exception as e:
        print(f"Database error: {e}")
        return []

def update_books_display(cabinet):
    """Update the text box with books from the selected cabinet."""
    books = get_books_in_cabinet(cabinet)
    book_list = "\n".join([f"\u2022 {book[0]}" for book in books]) if books else "No books found."

    # Update the text box content
    text_box.configure(state="normal")
    text_box.delete("0.0", "end")
    text_box.insert("0.0", f"Books in {cabinet}:\n\n{book_list}")
    text_box.configure(state="disabled")

    speak(f"Showing books in {cabinet}")

# Main UI Setup
app = ctk.CTk()
app.title("Cabinet")
app.geometry("600x400")
app.configure(bg="#2B2B2B")

# Use a grid layout
app.grid_columnconfigure(0, weight=1)  # Column for buttons
app.grid_columnconfigure(1, weight=3)  # Column for text box

# Header Label
label = ctk.CTkLabel(
    app,
    text="Select a Cabinet to Open:",
    font=("Helvetica", 18, "bold"),
    text_color="#E0E0E0"
)
label.grid(row=0, column=0, padx=10, pady=20, sticky="w")

# Buttons to select cabinets (column layout)
button1 = ctk.CTkButton(
    app,
    text="Open Cabinet 1",
    command=lambda: update_books_display("Cabinet 1"),
    fg_color="#3A3A3A",
    text_color="#FFFFFF"
)
button1.grid(row=1, column=0, padx=10, pady=10, sticky="w")

button2 = ctk.CTkButton(
    app,
    text="Open Cabinet 2",
    command=lambda: update_books_display("Cabinet 2"),
    fg_color="#3A3A3A",
    text_color="#FFFFFF"
)
button2.grid(row=2, column=0, padx=10, pady=10, sticky="w")

button3 = ctk.CTkButton(
    app,
    text="Open Cabinet 3",
    command=lambda: update_books_display("Cabinet 3"),
    fg_color="#3A3A3A",
    text_color="#FFFFFF"
)
button3.grid(row=3, column=0, padx=10, pady=10, sticky="w")

# Text Box to display books (aligned to the right)
text_box = ctk.CTkTextbox(
    app,
    width=400,
    height=300,
    font=("Courier", 14),
    fg_color="#1E1E1E",
    text_color="#A0A0A0"
)
text_box.grid(row=0, column=1, rowspan=4, padx=10, pady=10, sticky="nsew")
text_box.configure(state="disabled", wrap="word")

# Run the App
app.mainloop()
