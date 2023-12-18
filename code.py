# Import necessary libraries
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Caching mechanism for extracted features
cache = {}

# Image loading function
def load_image(file_path):
    # Check if the image is already in the cache
    if file_path in cache:
        return cache[file_path]

    # Read the image using OpenCV
    image = cv2.imread(file_path)
    cache[file_path] = image

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return hsv_image

# Feature extraction function using color histograms
def extract_features(image):
    # Calculate histograms for each channel (Hue, Saturation, Value)
    h_hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    s_hist = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()

    # Normalize the histograms
    cv2.normalize(h_hist, h_hist)
    cv2.normalize(s_hist, s_hist)
    cv2.normalize(v_hist, v_hist)

    # Concatenate the histograms into a single feature vector
    features = np.concatenate([h_hist, s_hist, v_hist])
    return features

# Process each database image
def process_image(db_path):
    # Load the database image and extract features
    db_image = load_image(db_path)
    db_features = extract_features(db_image)
    return db_features

# Search for similar images
def search_image(query_image_path, database_dir, top_n=5):
    # Load the query image and extract features
    query_image = load_image(query_image_path)
    query_features = extract_features(query_image)

    # Get paths of all images in the database
    database_paths = [os.path.join(database_dir, filename) for filename in os.listdir(database_dir)]

    # Use multithreading to process database images in parallel
    with ThreadPoolExecutor() as executor:
        database_features = list(executor.map(process_image, database_paths))

    # Calculate cosine similarity between the query image and database images
    similarities = cosine_similarity([query_features], database_features)[0]

    # Get indices of top N similar images
    most_similar_indices = np.argsort(similarities)[::-1][:top_n]

    # Return a list of tuples containing image paths and similarity scores
    return [(database_paths[i], similarities[i]) for i in most_similar_indices]

# Plot query image
def plot_image(image_path, target_size=(256, 256)):
    # Read and plot the query image using Matplotlib
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, target_size)
    plt.imshow(resized_image)
    plt.axis('off')
    plt.show()

# GUI class
class ImageSearchGUI:
    def __init__(self, root):
        # Initialize the GUI with a given root window
        self.root = root
        self.root.title("Image Search GUI")

        # Initialize paths for query image and database directory
        self.query_image_path = ""
        self.database_directory = ""

        # Create labels
        self.label_query = tk.Label(root, text="Query Image:")
        self.label_database = tk.Label(root, text="Database Directory:")

        # Create entry widgets
        self.entry_query = tk.Entry(root, width=40, state="readonly")
        self.entry_database = tk.Entry(root, width=40, state="readonly")

        # Create buttons
        self.button_browse_query = tk.Button(root, text="Browse", command=self.browse_query)
        self.button_browse_database = tk.Button(root, text="Browse", command=self.browse_database)
        self.button_search = tk.Button(root, text="Search", command=self.perform_search)

        # Create image display canvas
        self.canvas = tk.Canvas(root, width=400, height=400)

        # Grid layout
        self.label_query.grid(row=0, column=0, padx=10, pady=10)
        self.entry_query.grid(row=0, column=1, padx=10, pady=10)
        self.button_browse_query.grid(row=0, column=2, padx=10, pady=10)

        self.label_database.grid(row=1, column=0, padx=10, pady=10)
        self.entry_database.grid(row=1, column=1, padx=10, pady=10)
        self.button_browse_database.grid(row=1, column=2, padx=10, pady=10)

        self.button_search.grid(row=2, column=1, pady=20)
        self.canvas.grid(row=3, column=0, columnspan=3)

    def browse_query(self):
        # Open a file dialog to select the query image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # Update the query image path and display the image
            self.query_image_path = file_path
            self.entry_query.config(state="normal")
            self.entry_query.delete(0, tk.END)
            self.entry_query.insert(0, file_path)
            self.entry_query.config(state="readonly")
            self.display_image(file_path)

    def browse_database(self):
        # Open a file dialog to select the database directory
        dir_path = filedialog.askdirectory()
        if dir_path:
            # Update the database directory path
            self.database_directory = dir_path
            self.entry_database.config(state="normal")
            self.entry_database.delete(0, tk.END)
            self.entry_database.insert(0, dir_path)
            self.entry_database.config(state="readonly")

    def perform_search(self):
        # Perform image search and display the results
        if self.query_image_path and self.database_directory:
            results = search_image(self.query_image_path, self.database_directory, top_n=5)

            # Clear previous images from the canvas
            self.canvas.delete("all")

            # Display all matching images
            for path, similarity in results:
                self.display_image(path)
                print(f"Image: {path}, Similarity: {similarity}")

    def display_image(self, image_path):
        # Display an image on the canvas
        image = Image.open(image_path)
        image.thumbnail((400, 400))
        tk_image = ImageTk.PhotoImage(image)

        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image

if __name__ == "__main__":
    # Create a Tkinter root window and start the GUI
    root = tk.Tk()
    app = ImageSearchGUI(root)
    root.mainloop()
