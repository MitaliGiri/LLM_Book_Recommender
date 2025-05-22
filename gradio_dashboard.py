import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv('books_with_emotions.csv')

# Adding book covers as thumbnails (taking the largest ones)
books['large_thumbnail'] = books['thumbnail'] + "&fife=w800"

# Adding 'cover_not_found' image where cover na
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    "cover-not-found.jpg",
    books['large_thumbnail'],
)

# Code to create vector database
# Loading the document
raw_documents = TextLoader(file_path='tagged_description.txt', encoding='utf-8').load()

# Instantiating the text split up
text_splitter = CharacterTextSplitter(chunk_size= 0, chunk_overlap = 0, separator = '\n')
# 'chunk size' is set to 0 here so that we can be sure that text is split on separator rather than chunk size

# Applying text splitter to the documents
documents = text_splitter.split_documents(raw_documents)

# Create the document embedding and store them in vector database (use when needed)
db_books = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())



# Getting semantic recommendations
def retrieve_semantic_recommendations(
        query : str,
        category : str = None,
        tone : str = None,
        initial_top_k : int = 50,
        final_top_k : int = 16
) -> pd.DataFrame:

    # to get the recommendations against the database
    recs = db_books.similarity_search(query, k=initial_top_k)
    # getting isbns by splitting them off descriptions
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    # make dataframe of only of isbns that match
    book_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)


    # Applying filtering based on category
    if category != 'All' :
        book_recs = book_recs[book_recs['simple_categories'] == category].head(initial_top_k)
    else :
        book_recs = book_recs.head(final_top_k)


    # Sorting emotions based on probabilities
    if tone == 'Happy':
        book_recs.sort_values(by = 'joy', ascending = False, inplace = True)
    elif tone == 'Surprising':
        book_recs.sort_values(by = 'surprise', ascending = False, inplace = True)
    elif tone == 'Angry':
        book_recs.sort_values(by = 'anger', ascending = False, inplace = True)
    elif tone == 'Suspenseful':
        book_recs.sort_values(by = 'fear', ascending = False, inplace = True)
    elif tone == 'Sad':
        book_recs.sort_values(by = 'sadness', ascending = False, inplace = True)

    return book_recs


# Function that specifies what will be displayed on the dashboard
def recommend_books(
        query : str,
        category : str,
        tone : str
) :
    # Getting recommendations df from previous function
    recommendation = retrieve_semantic_recommendations(query, category, tone)
    results = []

    # Looping over all recommendations
    for _, row in recommendation.iterrows():
        description = row['description']
        truncated_desc_split = description.split()
        # Taking only first 30 words of description and then trailing it with '...' as limited space on dashboard
        truncated_description = " ".join(truncated_desc_split[:30]) + '...'


        # For displaying authors
        # Splitting authors on semicolon
        authors_split = row['authors'].split(';')

        # If we have 2 authors, joining them by and
        if len(authors_split)  == 2 :
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        # If there are more than 2 authors separating all with ',' except adding and before last
        elif len(authors_split) > 2 :
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        # If we have only one author
        else :
            authors_str = row['authors']

        # Displaying all this information under thumbnail as caption
        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row['large_thumbnail'], caption))

    return results

# For the dashboard
categories = ['All'] + sorted(books['simple_categories'].unique())
tones = ['All'] + ['Happy', 'Sad', 'Surprising', 'Angry', 'Suspenseful']

# Theme of gradio dashboard
with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    # Title of dashboard
    gr.Markdown("# Semantic Book Recommendation")

    # For user interaction
    with gr.Row():
        # To enter book description
        user_query = gr.Textbox(label = 'Please enter a description of a book : ',
                                placeholder = 'e.g. A story about forgiveness')
        # For selecting category
        category_dropdown = gr.Dropdown(choices = categories, label = 'Select a category : ', value = 'All')
        # For selecting emotion
        tone_dropdown = gr.Dropdown(choices = tones, label = 'Select an emotional tone : ', value = 'All')
        # Submit Button
        submit_button = gr.Button('Find Recommendations')


    # To display recommendations as a gallery
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = 'Recommended books', columns = 8, rows = 2)

    # Action to be done when submit button is clicked
    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)



if __name__ == '__main__':
    dashboard.launch()





