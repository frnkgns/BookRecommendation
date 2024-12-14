
import customtkinter
from customtkinter import *
import sqlite3
from PIL import Image
from subprocess import call
import pyttsx3
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

AttributeClicked = 0
con = sqlite3.connect("Data.db")
cur = con.cursor()

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
voices = engine.getProperty('voices', )

MainWindow = CTk()
#region Title Id
TitleID = [
    "21-1",
    "22-2",
    "22-1",
    "22-2",
    "22-3",
    "22-4",
    "22-5",
    "22-6",
    "22-7",
    "22-8",
    "22-9",
    "22-10",
    "22-11",
    "22-12",
    "22-13",
    "22-7",
    "23-1",
    "23-2",
    "23-3",
    "23-4",
    "23-5",
    "23-6",
    "23-7",
    "23-8",
    "23-9",
    "23-10",
    "23-11",
    "23-12",
    "23-13",
    "23-14",
    "23-15",
    "23-16",
    "23-17",
    "23-18",
    "23-19",
    "23-20",
    "23-21",
    "23-22",
    "23-23",
    "23-24",
    "23-25",
    "23-26",
    "23-27",
    "23-28",
    "23-29",
    "23-30",
    "23-31",
    "23-32",
    "23-33",
    "23-34",
    "23-35",
    "23-36",
    "23-37",
    "23-38",
    "23-39",
    "23-40",
    "23-41",
    "23-42",
    "23-43",
    "23-44",
    "23-45",
    "23-46",
    "23-47",
    "23-48",
    "23-49",
    "23-50",
    "23-51",
    "23-52",
    "23-53",
    "23-54",
    "23-55",
    "23-56",
    "23-57",
    "23-58",
    "23-59",
    "23-60",
    "23-61",
    "23-62",
    "23-63",
    "23-64",
    "23-65",
    "23-66",
    "23-67",
    "23-68",
    "23-69",
    "23-70",
    "23-71",
    "23-72",
    "23-73",
]
#endregion
#region Book Category List
Categories = [
    "Agriculture ",
    "Analytics ",
    "Artificial Intelligence ",
    "Automation ",
    "Business ",
    "Counseling ",
    "Data Analysis ",
    "E-Commerce ",
    "Education ",
    "Entertainment ",
    "Financial Management ",
    "Financial Monitoring ",
    "Geographic Information System (GIS) ",
    "Healthcare ",
    "Image Processing ",
    "IOT (Internet of Things) ",
    "Inventory ",
    "Lan-Based ",
    "Law Enforcement ",
    "Management System ",
    "Mobile App Development ",
    "Record Management ",
    "Renewable Energy ",
    "Scheduling System ",
    "Security ",
    "Social Services ",
    "Student Services ",
    "Transportation ",
    "Tracking System ",
    "Utilities ",
    "Web-Based "
]
#endregion
#region Technology Used List
Technologies = [
    "ajax",
    "android studio",
    "arduino",
    "arduino ide",
    "bootstrap",
    "c#",
    "c++",
    "codeigniter4",
    "css",
    "css3",
    "favicon",
    "gis",
    "html",
    "html5",
    "java",
    "javascript",
    "jquery",
    "kotlin",
    "mariadb",
    "mysql",
    "perl",
    "php",
    "python",
    "sublime text",
    "tomcat",
    "visual basic",
    "vs code",
    "xamp",
    "xml",
]
#endregion
Year = ["2021", "2022", "2023"]

# Window dimensions
WindowWidth = 1150
WindowHeight = 2500

# Get screen dimensions
ScreenWidth = MainWindow.winfo_screenwidth()
ScreenHeight = MainWindow.winfo_screenheight()

# Calculate x and y position for centering
x = int((ScreenWidth / 2) - (WindowWidth / 2))
y = int((ScreenHeight / 2) - (WindowHeight / 2))

# Set window geometry with centering
MainWindow.geometry(f"{WindowWidth}x{WindowHeight}+{x}+{y}")

# Make it fullscreen
MainWindow.attributes("-fullscreen", True)  # Use this for fullscreen mode

ProfileFrame = CTkFrame(MainWindow, height=400, width=300)
ProfileFrame.place(relx=0.01, rely=0.02)
StatisticFrame = CTkFrame(MainWindow, height=500, width=200)
StatisticFrame.place(relx=0.01, rely=0.42)

#region Show All Books By Year
def ShowAllByYear(e):
    AllBooks("Year")
    ChoseByCategoryText = customtkinter.StringVar(value="Category")
    ChoseByCategory.configure(variable=ChoseByCategoryText)
    ChoseByTechnologiesText = customtkinter.StringVar(value="Technology")
    ChoseByTechnologies.configure(variable=ChoseByTechnologiesText)
#endregion
#region Show All Books By Category
def ShowAllByCategory(e):
    AllBooks("Category")
    ChoseByYearText = customtkinter.StringVar(value="Year")
    ChoseByYear.configure(variable=ChoseByYearText)
    ChoseByTechnologiesText = customtkinter.StringVar(value="Technology")
    ChoseByTechnologies.configure(variable=ChoseByTechnologiesText)
#endregion
#region Show All Books By Tecnologies
def ShowAllByTechnologies(e):
    AllBooks("Technologies")
    ChoseByYearText = customtkinter.StringVar(value="Year")
    ChoseByYear.configure(variable=ChoseByYearText)
    ChoseByCategoryText = customtkinter.StringVar(value="Category")
    ChoseByCategory.configure(variable=ChoseByCategoryText)
#endregion
#region Go Back To Log In
def GoBackToLogIn():
    MainWindow.destroy()
    call(["python", "LogInForm.py"])
#endregion

Searchbar = CTkEntry(MainWindow, placeholder_text="Search Title", width=530)
Searchbar.place(relx=0.165, rely=0.02)
ChoseByYearText = customtkinter.StringVar(value="Year")
ChoseByYear = CTkOptionMenu(MainWindow, values=Year, variable=ChoseByYearText,
                            dynamic_resizing=False, command=ShowAllByYear)
ChoseByYear.place(relx=0.558, rely=0.02)
ChoseByCategoryText = customtkinter.StringVar(value="Category")
ChoseByCategory = CTkOptionMenu(MainWindow, values=Categories, variable=ChoseByCategoryText,command=ShowAllByCategory)
ChoseByCategory.place(relx=0.668, rely=0.02)

ChoseByTechnologiesText = customtkinter.StringVar(value="Technology")
ChoseByTechnologies = CTkOptionMenu(MainWindow, values=Technologies, variable=ChoseByTechnologiesText,
                                    command=ShowAllByTechnologies)
ChoseByTechnologies.place(relx=0.78, rely=0.02)

MultiPurposeBtn = CTkButton(MainWindow, text="Exit", fg_color="red", hover_color="darkred",
                            command=lambda: GoBackToLogIn())
MultiPurposeBtn.place(relx=0.89, rely=0.02)

# -------------Student Account------------------------------------------------
with open("LogInID", "r+") as LogInID:
    GetLoginID = LogInID.readline()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(Categories)


#region TF-IDF, Cosine Similarity and Matrix Precision, Recall and F1-Score
def CombinedFunction():
    global query, titles
    ExistedTitleFrame = {}
    ExistedTitleLabel = {}
    index = 0
    row = 0
    column = 0

    # Step 1: Retrieve User Data and Preferences
    query = f"""
        select * from accountinfo
        where id = '{GetLoginID}'
    """
    cur.execute(query)
    Accountinfo = cur.fetchall()
    Accountinfolist = [e for e in Accountinfo]

    # Extract user information
    ID = Accountinfolist[0][0]
    Name = "Name: " + Accountinfolist[0][1] + " " + Accountinfolist[0][2]
    Gender = "Gender: " + Accountinfolist[0][3]
    Course = "Course: " + Accountinfolist[0][4] + "-" + str(Accountinfolist[0][5]) + ": " + Accountinfolist[0][6]
    Category = "Likes: " + Accountinfolist[0][7]

    # Display user info in the UI
    DisplayID = CTkLabel(ProfileFrame, font=("", 20), text=ID)
    DisplayID.place(relx=0.3, rely=0.05)
    DisplayName = CTkLabel(ProfileFrame, text=Name, justify="left", wraplength=150)
    DisplayName.place(relx=0.1, rely=0.2)
    DisplayGender = CTkLabel(ProfileFrame, text=Gender, justify="left", wraplength=150)
    DisplayGender.place(relx=0.1, rely=0.35)
    DisplayCourse = CTkLabel(ProfileFrame, text=Course, justify="left", wraplength=150)
    DisplayCourse.place(relx=0.1, rely=0.5)
    DisplayCategory = CTkLabel(ProfileFrame, text=Category, justify="left", wraplength=150)
    DisplayCategory.place(relx=0.1, rely=0.7)

    # Step 2: Retrieve Book Data
    query_books = """
        select * from bookinfo
    """
    cur.execute(query_books)
    Books = cur.fetchall()

    # Step 3: Extract user preferences (likes) and combine with books data for TF-IDF vectorization
    user_likes = Accountinfolist[0][7]  # User's interests or likes as a string
    user_likes_list = user_likes.split(",")  # Split interests by comma if they are separated
    user_searched = Accountinfolist[0][8]

    courseList = {"BSIT", "BSCS"}
    category_weight = 5  # Apply a weight of 5 for each category
    technologies_weight = 4  # Apply a weight of 4 for each technology (you can adjust this as needed)
    course_weight = 5

    # Function to apply category and technology weights to book descriptions
    def apply_weights(corpus, category_weight, technology_weight, course_weight):
        weighted_corpus = []
        for text in corpus:
            # Ensure that 'text' is a string (if it's a list, join it into a string)
            if isinstance(text, list):
                text = " ".join(text)  # Join the list into a single string

            weighted_text = text
            # Apply category weights
            for category in Categories:
                if category.lower() in text.lower():  # Check if category is mentioned in the text
                    weighted_text += " " + (category * category_weight)  # Add weighted category

            # Apply technology weights
            for technology in Technologies:
                if technology.lower() in text.lower():  # Check if technology is mentioned in the text
                    weighted_text += " " + (technology * technology_weight)  # Add weighted technology

            # Apply course weights
            for course in courseList:
                if course.lower() in text.lower():  # Check if course is mentioned in the text
                    weighted_text += " " + (course * course_weight)  # Add weighted course

            weighted_corpus.append(weighted_text)
        return weighted_corpus

    # Combine book titles, types, and authors into a corpus for vectorization
    book_texts = []
    for book in Books:
        book_id = book[0]  # Assuming book[0] contains the title id
        book_title = book[2]  # titles
        book_type = book[3]  # category
        book_technologies = book[4]  # technologies
        book_course = book[5]  # course
        book_description = book[7]  # course
        combined_text = f"{book_title} {book_type} {book_course} {book_technologies} {book_description}"
        book_texts.append((book_id, combined_text))  # Store book id along with the combined text

    corpus = [user_likes_list + [user_searched]] + [text for _, text in book_texts]  # User's preferences + book combined text

    # Apply category and technology weights to the corpus
    weighted_corpus = apply_weights(corpus, category_weight, technologies_weight, course_weight)

    # Step 4: Apply TF-IDF Vectorization and calculate cosine similarity
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2)  # Adjusted for better results
    tfidf_matrix = vectorizer.fit_transform(weighted_corpus)

    # Get the user's preference vector (first entry in the corpus)
    user_vector = tfidf_matrix[0]

    # Get the book vectors (all other entries in the corpus)
    book_vectors = tfidf_matrix[1:]

    # Step 5: Calculate cosine similarity between user's preferences and each book
    similarity_scores = cosine_similarity(user_vector, book_vectors).flatten()

    # Add a threshold to avoid including books with zero or near-zero similarity
    similarity_threshold = 0.05
    similarity_threshold_predicted = 0.2
    relevant_books = [book_texts[i][0] for i in range(len(similarity_scores)) if similarity_scores[i] > similarity_threshold]
    predicted_books = [book_texts[i][0] for i in range(len(similarity_scores)) if similarity_scores[i] > similarity_threshold_predicted]

    print(f"User Id: {ID} \nUser Preference \n{Category} \nUser Description: {user_searched}\n")
    # print("Relevant Books:", relevant_books)
    # print("Predicted Books:", predicted_books)

    # Sort books by similarity scores (higher scores mean better match)
    sorted_indices = np.argsort(similarity_scores)[::-1]  # Descending order
    sorted_books = [(book_texts[i][0], Books[i][2], similarity_scores[i]) for i in
                    sorted_indices]  # (Book ID, Book Title, Similarity Score)

    # Step 7: Display Recommendations in the UI
    RecommendedBookFrame = CTkScrollableFrame(MainWindow, height=270, width=750,
                                              label_text="RECOMMENDED FOR YOU",
                                              label_anchor="w",
                                              label_font=("Consolas", 30, "bold"),
                                              label_fg_color="transparent")
    RecommendedBookFrame.place(relx=0.165, rely=0.07)
    BookCover = CTkImage(light_image=Image.open("Book Front Cover.png"),
                         dark_image=Image.open("Book Front Cover.png"),
                         size=(165, 300))

    FrameHasElements = RecommendedBookFrame.winfo_children()
    if FrameHasElements:
        for widget in RecommendedBookFrame.winfo_children():
            widget.destroy()

    # Display top 3 book recommendations
    print("Top 10 Book Recommendations based on User Preferences:")
    for i, (book_id, book_title, score) in enumerate(sorted_books[:10]):
        print(f"{i + 1}. Book ID: {book_id}, Title: {book_title}, Similarity Score: {score:.2f}")

        row = i // 4  # Divide into rows (adjust 5 to control number of items per row)
        column = i % 4  # Ensure no more than 5 items per row

        ExistedTitleFrame[book_title] = CTkFrame(RecommendedBookFrame, height=280, width=150, fg_color="transparent")
        ExistedTitleFrame[book_title].grid(column=column, row=row, padx=20, pady=10, sticky="w")
        ExistedTitleLabel[book_title] = CTkLabel(ExistedTitleFrame[book_title], font=("Times", 14), text=book_title,
                                                 image=BookCover, wraplength=100)
        ExistedTitleLabel[book_title].place(relx=-0.02, rely=-0.02)
        ExistedTitleLabel[book_title].bind("<Button-1>",
                                           lambda e, bookid=book_id, booktitle=book_title: ShowBookInfo(bookid,
                                                                                                        booktitle))

    # Step 6: Calculate Precision, Recall, and F1 Score with zero_division handling
    if len(relevant_books) > 0 and len(predicted_books) > 0:
        # True positives: Books that are both relevant and predicted
        true_positives = len([book for book in predicted_books if book in relevant_books])

        # Precision: Relevant books recommended / Total recommended books
        precision = true_positives / len(predicted_books) if len(predicted_books) > 0 else 0

        # Recall: Relevant books recommended / Total relevant books
        recall = true_positives / len(relevant_books) if len(relevant_books) > 0 else 0

        # F1 Score: Harmonic mean of precision and recall
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0  # To handle the case where both precision and recall are 0

        print(f"\nTF-IDF Matrix")
        print(f"Threshold: (Relevant Books: 0.05) (Recommended Books; 0.2")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print("Relevant Books:", relevant_books)
        print("Predicted Books:", predicted_books)

    else:
        print("No relevant or predicted books to calculate precision, recall, and F1 score.")
#endregion

#region Bert Based Recommendation
def BertBasedRecommendation():
    global query, titles

    # Step 1: Retrieve User Data and Preferences (same as before)
    query = f"""
        select * from accountinfo
        where id = '{GetLoginID}'
    """
    cur.execute(query)
    Accountinfo = cur.fetchall()
    Accountinfolist = [e for e in Accountinfo]

    ID = Accountinfolist[0][0]
    user_likes = Accountinfolist[0][7]
    user_likes_list = user_likes.split(",")
    user_searched = Accountinfolist[0][8]
    user_course = Accountinfolist[0][4]

    query_books = """
        select * from bookinfo
    """
    cur.execute(query_books)
    Books = cur.fetchall()

    # Combine book data into a text corpus
    book_texts = []
    for book in Books:
        book_id = book[0]
        book_title = book[2]
        book_type = book[3]
        book_technologies = book[4]
        book_course = book[5]
        book_description = book[7]
        combined_text = f"{book_title} {book_type} {book_course} {book_technologies} {book_description}"
        book_texts.append((book_id, combined_text))

    # Create the corpus including user preferences
    corpus = [" ".join(user_likes_list) + " " + user_searched + " " + user_course] + [text for _, text in book_texts]

    # Step 2: Use BERT for text embeddings
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def encode_texts(texts):
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
          #  print(f"Tokenized input: {inputs}")  # Debugging: print tokenized input
            outputs = model(**inputs)
          #  print(f"Model Output: {outputs.last_hidden_state.shape}")  # Debugging: check model output shape
            embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy()[0])
        return np.array(embeddings)

    # Encode user preferences and book texts
    embeddings = encode_texts(corpus)
    user_vector = embeddings[0]
    book_vectors = embeddings[1:]

    # Step 3: Calculate cosine similarity
    similarity_scores = cosine_similarity([user_vector], book_vectors).flatten()
    # print(f"Similarity Scores: {similarity_scores[:10]}")  # Debugging: print first 10 similarity scores

    # Add thresholding for relevant and predicted books
    Relevant_Books_to_the_User_Profile = 0.7
    Recommended_Books_Score_Of = 0.8
    relevant_books2 = [book_texts[i][0] for i in range(len(similarity_scores)) if similarity_scores[i] > Relevant_Books_to_the_User_Profile]
    predicted_books2 = [book_texts[i][0] for i in range(len(similarity_scores)) if similarity_scores[i] > Recommended_Books_Score_Of]

    # Step 4: Precision, Recall, and F1 Score
    if len(relevant_books2) > 0 and len(predicted_books2) > 0:
        true_positives = len([book for book in predicted_books2 if book in relevant_books2])
        precision = true_positives / len(predicted_books2) if len(predicted_books2) > 0 else 0
        recall = true_positives / len(relevant_books2) if len(relevant_books2) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        print(f"\nBERT Matrix")
        print(f"Threshold: (Relevant Books: 0.7) (Recommended Books; 0.8")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
        # Debugging: check relevant and predicted books
        print(f"Relevant Books: {relevant_books2}")
        print(f"Predicted Books: {predicted_books2}\n")
    else:
        print("No relevant or predicted books to calculate precision, recall, and F1 score.")

    # Step 5: Display Top Recommendations
    sorted_indices = np.argsort(similarity_scores)[::-1]
    sorted_books = [(book_texts[i][0], Books[i][2], similarity_scores[i]) for i in sorted_indices]

    print("Top 10 Book Recommendations based on BERT:")
    for i, (book_id, book_title, score) in enumerate(sorted_books[:10]):
        print(f"{i + 1}. Book ID: {book_id}, Title: {book_title}, Similarity Score: {score:.2f}")
#endregion

#region Cabinet Query
def CabinetQuery(cabinetNumber):
    if cabinetNumber == 1:
        AllBooks(1)
    elif cabinetNumber == 2:
        AllBooks(2)

    elif cabinetNumber == 3:
        AllBooks(3)
#endregion
#region All Books
def AllBooks(Mode):
    global query, titles
    ExistedTitleFrame = {}
    ExistedTitleLabel = {}
    index = 0
    row = 0
    column = 0
    FirstRun = True
    BookCover = CTkImage(light_image=Image.open("Book Front Cover.png"),
                         dark_image=Image.open("Book Front Cover.png"),
                         size=(165, 300))

    ExistedTitleFrame.clear()
    ExistedTitleLabel.clear()
    RecommendedBookFrame = CTkScrollableFrame(MainWindow, height=550, width=800,
                                              label_text="ALL BOOKS",
                                              label_font=("Consolas", 30, "bold"),
                                              label_fg_color="transparent",
                                              label_anchor="w")
    RecommendedBookFrame.place(relx=0.165, rely=0.50)
    cabinetBtnFrame = CTkFrame(MainWindow, height=100, width=800)
    cabinetBtnFrame.place(relx=0.300, rely=0.50)
    cabinetLbl = CTkLabel(cabinetBtnFrame, text="Total: ")
    cabinetLbl.grid(row=0, column=0, padx=10, pady=10, sticky="w")
    cabinet1 = CTkButton(cabinetBtnFrame, text="Cabinet 1", command=lambda: CabinetQuery(1))
    cabinet1.grid(row=0, column=1, padx=10, pady=10, sticky="w")
    cabinet2 = CTkButton(cabinetBtnFrame, text="Cabinet 2", command=lambda: CabinetQuery(2))
    cabinet2.grid(row=0, column=2, padx=10, pady=10, sticky="w")
    cabinet3 = CTkButton(cabinetBtnFrame, text="Cabinet 3", command=lambda: CabinetQuery(3))
    cabinet3.grid(row=0, column=3, padx=10, pady=10, sticky="w")
    cabinetBtnFrame.lift()

    FrameHasElements = RecommendedBookFrame.winfo_children()
    if FrameHasElements:
        for elements in RecommendedBookFrame.winfo_children():
            elements.destroy()

    if Mode == "All":
        query = f"""
                    select * from bookinfo
                """
    elif Mode == "Year":
        SelectedYear = ChoseByYear.get()
        RecommendedBookFrame.configure(label_text=f"ALL BOOKS YEAR {SelectedYear.upper()}")
        FirstRun = False
        query = f"""
                    select * from bookinfo 
                    where year = '{SelectedYear}'
                """
    elif Mode == "Category":
        SelectedCategory = ChoseByCategory.get()
        selectedcategory = SelectedCategory.strip()
        RecommendedBookFrame.configure(label_text=f"ALL TITLES IN {selectedcategory.upper()} CATEGORY")
        FirstRun = False
        query = f"""
                    select * from bookinfo
                    where category like '%{selectedcategory}%'
                """
    elif Mode == "Technologies":
        SelectedTechnologies = ChoseByTechnologies.get()
        RecommendedBookFrame.configure(label_text=f"ALL TITLES THAT USES {SelectedTechnologies.upper()}")
        FirstRun = False
        query = f"""
                    select * from bookinfo
                    where technologies like '%{SelectedTechnologies}%'
                """
    elif Mode == "Search":
        SelectTitle = Searchbar.get()
        FirstRun = False
        if len(SelectTitle) >= 1:
            RecommendedBookFrame.configure(label_text=f"ALL TITLES THAT MATCHES {SelectTitle.upper()}")
            query = f"""
                    select * from bookinfo
                    where Titles like '%{SelectTitle}%'
                    """
        else:
            engine.say("you haven't entered something, search titles to continue")
            engine.runAndWait()
            # AllBooks("All")

    elif Mode == 1:
        FirstRun = False
        RecommendedBookFrame.configure(label_text=f"ALL TITLES")
        query = f"""
                    select * from bookinfo
                    where Location like '%Cabinet 1%' and Location like '%Unpublished Material%'
                """

    elif Mode == 2:
        FirstRun = False
        RecommendedBookFrame.configure(label_text=f"ALL TITLES")
        query = f"""
                    select * from bookinfo
                    where Location like '%Cabinet 2%' and Location like '%Unpublished Material%'
                """

    elif Mode == 3:
        FirstRun = False
        RecommendedBookFrame.configure(label_text=f"ALL TITLES")
        query = f"""
                    select * from bookinfo
                    where Location like '%Cabinet 3%' and Location like '%Unpublished Material%'
                """



    cur.execute(query)
    TitleInfo = cur.fetchall()
    TitleInfolist = [e for e in TitleInfo]

    if len(TitleInfolist) > 0:
        for Title in TitleInfolist:
            titles = TitleInfolist[index][2]
            BookID = TitleInfolist[index][0]

            if column != 4:
                ExistedTitleFrame[Title] = CTkFrame(RecommendedBookFrame, height=285, width=150, fg_color="transparent")
                ExistedTitleFrame[Title].grid(column=column, row=row, padx=20, pady=10, sticky="w")
                ExistedTitleLabel[Title] = CTkLabel(ExistedTitleFrame[Title],
                                                    font=("Times New Roman", 14),
                                                    text=titles,
                                                    image=BookCover,
                                                    wraplength=100)
                ExistedTitleLabel[Title].place(relx=-0.02, rely=-0.02)
                ExistedTitleLabel[Title].bind("<Button-1>",
                                              lambda e, bookid=BookID, TiTles=titles:
                                              ShowBookInfo(bookid, TiTles))

                column += 1
            else:
                row += 1
                column = 0

                ExistedTitleFrame[Title] = CTkFrame(RecommendedBookFrame, height=285, width=150, fg_color="transparent")
                ExistedTitleFrame[Title].grid(column=column, row=row, padx=20, pady=10, sticky="w")
                ExistedTitleLabel[Title] = CTkLabel(ExistedTitleFrame[Title],
                                                    font=("Times New Roman", 14),
                                                    text=titles,
                                                    image=BookCover,
                                                    wraplength=100)
                ExistedTitleLabel[Title].place(relx=-0.02, rely=-0.02)
                ExistedTitleLabel[Title].bind("<Button-1>",
                                              lambda e, bookid=BookID, TiTles=titles:
                                              ShowBookInfo(bookid, TiTles))
                column += 1
            index += 1
    else:
        FirstRun = True
        engine.say("We don't have data yet that matches your preference")
        engine.runAndWait()
        # AllBooks("All")

    EngineSays(FirstRun, index, Mode, cabinetLbl)
#endregion
#region Text to Speech Engine
def EngineSays(FirstRun, numberoftitles, Mode, cabinetTxt):
    if FirstRun == True:
        pass
    else:
        if Mode == "Year":
            engine.say("I found " + str(numberoftitles) + " titles year " + ChoseByYear.get())
            cabinetTxt.configure(text= f"Total {str(numberoftitles)}")

        elif Mode == "Category":
            engine.say("I found " + str(numberoftitles) + ChoseByCategory.get() + " titles ")
            cabinetTxt.configure(text= f"Total {str(numberoftitles)}")

        elif Mode == "Technologies":
            engine.say("I found " + str(numberoftitles) + " that uses " + ChoseByTechnologies.get())
            cabinetTxt.configure(text= f"Total {str(numberoftitles)}")

        elif Mode == "Search":
            if len(Searchbar.get()) > 1:
                engine.say("I found " + str(numberoftitles) + " that matches " + Searchbar.get())
                cabinetTxt.configure(text=f"Total {str(numberoftitles)}")

        elif Mode == 1:
            engine.say("I found " + str(numberoftitles) + " in cabinet 1 base on your preference")
            cabinetTxt.configure(text= f"Total {str(numberoftitles)}")

        elif Mode == 2:
            engine.say("I found " + str(numberoftitles) + " in cabinet 2 base on your preference")
            cabinetTxt.configure(text= f"Total {str(numberoftitles)}")

        elif Mode == 3:
            engine.say("I found " + str(numberoftitles) + " in cabinet 3 base on your preference")
            cabinetTxt.configure(text= f"Total {str(numberoftitles)}")

        engine.runAndWait()
    #here we can do the precision, recall and f1 score
#endregion
#region Show Book Info
def ShowBookInfo(BookID, Booktitle):

    if len(BookID) > 1:
        engine.say(" Opening title id " + BookID)
        engine.runAndWait()
    else:
        engine.say(" Opening title ")
        engine.runAndWait()

    ChoseByYear.configure(state='disabled')
    ChoseByCategory.configure(state='disabled')
    ChoseByTechnologies.configure(state='disabled')

    titlecollectionframe = CTkFrame(MainWindow, height=720, width=775)
    titlecollectionframe.place(relx=0.165, rely=0.07)
    BookCover = CTkImage(light_image=Image.open("Book Front Cover.png"),
                         dark_image=Image.open("Book Front Cover.png"),
                         size=(365, 600))
    BookLabel = CTkLabel(titlecollectionframe, font=("Times New Roman", 25),
                         text=Booktitle, image=BookCover, wraplength=200)
    BookLabel.place(relx=0.05, rely=0.1)

    Hidebtn = CTkButton(titlecollectionframe, text="BACK", command=lambda: EnableSelection())
    Hidebtn.place(relx=0.75, rely=0.88)

    FrameHasElements = titlecollectionframe.winfo_children()
    if FrameHasElements:
        for widget in titlecollectionframe.winfo_children():
            widget.destroy

    query = f"""
                    select * from bookinfo
                    where titles = '{Booktitle}'
                """
    cur.execute(query)
    BookInfo = cur.fetchall()
    bookinfolist = [e for e in BookInfo]

    # print(bookinfolist)
    ID = CTkLabel(titlecollectionframe, text="Title ID: " + str(bookinfolist[0][0]),
                  font=("Consolas", 20), justify="left", wraplength=300)
    ID.place(relx=0.55, rely=0.2)
    course = CTkLabel(titlecollectionframe, text="Course: " + str(bookinfolist[0][5]),
                      font=("Consolas", 20), justify="left", wraplength=300)
    course.place(relx=0.55, rely=0.25)
    year = CTkLabel(titlecollectionframe, text="Year Published: " + str(bookinfolist[0][1]),
                    font=("Consolas", 20), justify="left", wraplength=300)
    year.place(relx=0.55, rely=0.3)
    category = CTkLabel(titlecollectionframe, text="Category: " + str(bookinfolist[0][3]),
                        font=("Consolas", 20), justify="left", wraplength=300)
    category.place(relx=0.55, rely=0.35)
    technology = CTkLabel(titlecollectionframe, text="Technology Use: " + str(bookinfolist[0][4]),
                          font=("Consolas", 20), justify="left", wraplength=300)
    technology.place(relx=0.55, rely=0.43)
    location = CTkLabel(titlecollectionframe, text="Location: " + str(bookinfolist[0][6]),
                        font=("Consolas", 20), justify="left", wraplength=300)
    location.place(relx=0.55, rely=0.51)
    authors = CTkLabel(titlecollectionframe, text="Authors: " + str(bookinfolist[0][7]),
                       font=("Consolas", 20), justify="left", wraplength=300)
    authors.place(relx=0.55, rely=0.61)

    def EnableSelection():
        titlecollectionframe.destroy()
        ChoseByYear.configure(state='normal')
        ChoseByCategory.configure(state='normal')
        ChoseByTechnologies.configure(state='normal')

    query = f"""
        select nobookviewed, recentlyviewed from accountstatistics
        where id = '{GetLoginID}'
    """
    cur.execute(query)
    Accountinfo = cur.fetchall()
    accountinfolist = [e for e in Accountinfo]
    noofbookviewed = accountinfolist[0][0]
    recentlyviewed = bookinfolist[0][2]

    addviewbooks = noofbookviewed + 1

    query = f"""
        update accountstatistics
        set nobookviewed = '{addviewbooks}', recentlyviewed = '{recentlyviewed}'
        where id = '{GetLoginID}'
    """
    cur.execute(query)
    con.commit()

    query = f"""
        select noofviews from bookinfo
    """
    cur.execute(query)
    Noofviews = cur.fetchall()[0][0]
    newviews = int(Noofviews) + 1

    query = f"""
        update bookinfo
        set noofviews = '{newviews}'
        where "title id" = '{BookID}'
    """
    cur.execute(query)
    con.commit()

    MostViedBooks()
    ShoRecentViewed()
#endregiono
#region Recent View
def ShoRecentViewed():
    global Booktitle
    RecentlyViewed = {}

    query = f"""
        select recentlyviewed from accountstatistics
        where id = '{GetLoginID}'
    """
    cur.execute(query)
    Booktitle = cur.fetchall()[0][0]

    FrameLabel = CTkLabel(StatisticFrame, text= "RECENTLY VIEWED", font= ("Consolas", 20, "bold"))
    FrameLabel.place(relx= 0.1, rely= 0.05)
    BookCover = CTkImage(light_image=Image.open("Book Front Cover.png"),
                         dark_image=Image.open("Book Front Cover.png"),
                         size=(165, 300))
    BookLabel = CTkLabel(StatisticFrame, font=("Times New Roman", 15),
                         text= Booktitle, image=BookCover, wraplength=100)
    BookLabel.place(relx=0.1, rely=0.15)
    BookLabel.bind("<Button-1>", lambda e, bookid= None, booktitle= Booktitle: ShowBookInfo(bookid, booktitle))
#endregion
#region Most View Books
def MostViedBooks():
    global query, titles
    ExistedTitleFrame = {}
    ExistedTitleLabel = {}
    index = 0
    row = 0
    column = 0

    RecommendedBookFrame = CTkScrollableFrame(MainWindow, height=650, width=330,
                                              label_text="TRENDING TITLES",
                                              label_font=("Consolas", 30, "bold"),
                                              label_fg_color="transparent",
                                              label_anchor="w")
    RecommendedBookFrame.place(relx=0.735, rely=0.07)
    BookCover = CTkImage(light_image=Image.open("Book Front Cover.png"),
                         dark_image=Image.open("Book Front Cover.png"),
                         size=(165, 300))
    FrameHasElements = RecommendedBookFrame.winfo_children()
    if FrameHasElements:
        for widget in RecommendedBookFrame.winfo_children():
            widget.destroy()

    query = f"""
            select "title id", titles, noofviews
            from bookinfo
            order by noofviews desc
            limit 10
        """

    cur.execute(query)
    TitleInfo = cur.fetchall()
    TitleInfolist = [e for e in TitleInfo]
    # print(TitleInfolist)

    rank = 1
    if index <= 9:
        for Title in TitleInfolist:
            titles = TitleInfolist[index][1]
            BookID = TitleInfolist[index][0]
            noofviews = TitleInfolist[index][2]
            RankText = "#" + str(rank) + "\nViews: " + str(noofviews)

            ExistedTitleFrame[Title] = CTkFrame(RecommendedBookFrame, height=300, width=150, fg_color="transparent")
            ExistedTitleFrame[Title].grid(column=0, row=row, padx=10, pady=10, sticky="w")
            ExistedTitleLabel[Title] = CTkLabel(ExistedTitleFrame[Title], font=("Times", 14), text=titles,
                                                image=BookCover,
                                                wraplength=100)
            ExistedTitleLabel[Title].place(relx=-0.02, rely=-0.02)
            ExistedTitleLabel[Title].bind("<Button-1>",
                                          lambda e, bookid=BookID, TiTles=titles:
                                          ShowBookInfo(bookid, TiTles))
            NumberRank = CTkLabel(RecommendedBookFrame, text=RankText, font=("Consolas", 20),
                                  justify="left")
            NumberRank.grid(column=2, row=row)
            row += 1

            index += 1
            rank += 1
#endregion
#region RecommendedBooks
#endregion

AllBooks("All")
CombinedFunction()
BertBasedRecommendation()
MostViedBooks()
ShoRecentViewed()

MainWindow.bind('<Return>', lambda e: AllBooks("Search"))
MainWindow.mainloop()
