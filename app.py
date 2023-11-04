from flask import Flask, render_template, request, session, redirect
import pickle

import numpy as np
import pandas as pd

ratings_df = pd.read_csv("./notebook/Ratings.csv")

# Top 50 popular books with num of ratings and avg ratings
popular_df = pd.read_pickle("popular_df.pkl")
# popular books with num of ratings >100 and avg ratings
popular_df_1 = pd.read_pickle("popular_df_1.pkl")
# popular books with num of ratings and avg ratings
popular_df_0 = pd.read_pickle("popular_df_0.pkl")
# pivot table of users as columns and movies as index
pt = pd.read_pickle("pt.pkl")
# Books Dataframe
books_df = pd.read_pickle("books_df.pkl")
# Similarity score using cosine similarity for pivot table
cos_sim_matrix = pd.read_pickle("similarity_scores.pkl")
# books which has atleast 50 ratings and users who read more than 200 books
final_ratings = pd.read_pickle("final_ratings.pkl")

merged_df = ratings_df.merge(books_df, on='ISBN')


app = Flask(__name__)
app.secret_key = 'your_secret_key'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/set_ticker', methods=['POST'])
def set_ticker():
    user_id = int(request.form.get('user_id'))
    session['user_id'] = user_id
    return redirect('/user')


@app.route('/popular')
def popular():
    return render_template('popular.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )


def recommend(book_name):
    author = books_df[books_df['Book-Title']
                      == book_name]['Book-Author'].tolist()[0]
    author_rec = author_based_recommender(author)

    if book_name not in pt.index.values:
        return author_rec, author

    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(
        list(enumerate(cos_sim_matrix[index])), key=lambda x: x[1], reverse=True)[1:11]

    data = []
    for i in similar_items:
        item = []
        temp_df = books_df[books_df['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates(
            'Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates(
            'Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates(
            'Book-Title')['Image-URL-M'].values))

        data.append(item)
    return data, author_rec, author


def author_based_recommender(author_name):
    author_books = popular_df_1[popular_df_1['Book-Author'] == author_name]
    author_books = author_books.drop_duplicates(
        subset=["Book-Title", "Publisher"], keep='first')
    top_books = author_books.head(5)
    if top_books.empty:
        author_books = popular_df_0[popular_df_0['Book-Author'] == author_name]
        author_books = author_books.drop_duplicates(
            subset=["Book-Title", "Publisher"], keep='first')
        sorted_books = author_books.sort_values(
            by=['num_ratings', 'avg_rating'], ascending=False)
        top_books = sorted_books.head(5)

    return top_books


count = 0


def user_rec(user_id, user_data):
    unique_titles = user_data['Book-Title'].unique()
    filtered_titles = [title for title in unique_titles if title in pt.index]
    all_title_data = []
    all_author_data = []

    for title in unique_titles:
        title_data = None
        author_data = None
        subset_df = user_data[user_data['Book-Title'] == title]
        if not subset_df.empty:
            if title in filtered_titles:
                title_data, author_data, author = recommend(title)
                if user_id in final_ratings['User-ID'].values:
                    title_data = title_data[:1]
                    author_data = author_data.head(1)
            else:
                author = books_df[books_df['Book-Title']
                                  == title]['Book-Author'].tolist()[0]
                author_data = author_based_recommender(author)

        if title_data is not None:
            all_title_data.append(title_data)
        if author_data is not None:
            all_author_data.append(author_data)

    combined_title_data_df = pd.DataFrame([book for title_data in all_title_data for book in title_data], columns=[
                                          'Book-Title', 'Book-Author', 'Image-URL-M'])
    combined_author_data_df = pd.concat(all_author_data, ignore_index=True)
    print(combined_author_data_df.shape)

    return combined_title_data_df, combined_author_data_df


@app.route('/recommender', methods=['GET', 'POST'])
def title_rec():
    title_input = request.form.get('Book_Title')
    if title_input == None:
        title_input = "1984"
    data = []  # Initialize data as an empty list
    author_data = None
    author_name = None

    if books_df['Book-Title'].str.contains(title_input).any():
        try:
            result = recommend(title_input)
            print(len(result))
            if len(result) > 2:
                data = result[0]
                author_data = result[1]
                author_name = result[2]
                author_record_empty = True
            else:
                author_record = result[0]
                print(author_record.shape)
                author_name = result[1]
                author_record_empty = author_record.empty
                return render_template("title_rec.html", title_input=title_input, author_name=author_name, author_record=author_record, author_record_empty=author_record_empty,
                                       book_name=list(
                                           author_record['Book-Title'].values),
                                       author=list(
                                           author_record['Book-Author'].values),
                                       image=list(author_record['Image-URL-M'].values),)
        except IndexError:
            print("Enter the Title which is available")
            not_found_message = "Please Enter the Title which is available."
            return render_template('not_found.html', not_found_message=not_found_message)

    return render_template("title_rec.html", title_input=title_input, data=data, author_name=author_name, author_record_empty=author_record_empty,
                           book_name=list(author_data['Book-Title'].values),
                           author=list(author_data['Book-Author'].values),
                           image=list(author_data['Image-URL-M'].values),)


@app.route('/author', methods=['GET', 'POST'])
def author_rec():
    author_input = request.form.get('author_name')
    if author_input == None:
        author_input = "Scott Turow"
    if books_df['Book-Author'].str.contains(author_input).any():
        try:
            data = author_based_recommender(author_input)
        except IndexError:
            print("Enter the Author Who is available")
            not_found_message = "Please Enter the Author Who is available."
            return render_template('not_found.html', not_found_message=not_found_message)

    return render_template("author_rec.html", author_input=author_input,
                           book_name=list(data['Book-Title'].values),
                           author=list(data['Book-Author'].values),
                           image=list(data['Image-URL-M'].values),)


# def show_image(val):
#     return '<img src="{}" width=50></img'.format(val)


@app.route('/user', methods=['GET', 'POST'])
def user_history():

    try:
        user_id = session.get('user_id')

        user_data = merged_df[merged_df['User-ID'] ==
                              user_id].sort_values(by="Book-Rating", ascending=False)
        user_data = user_data[['Book-Rating',
                               'Book-Title', 'Book-Author', 'Image-URL-M']]
        user_data = user_data.reset_index(drop=True)
        title_rec, auth_rec = user_rec(user_id, user_data)
    except TypeError:
        print("Enter the User Who is available")
        not_found_message = "Please Enter the User Who is available."
        return render_template('not_found.html', not_found_message=not_found_message)

    return render_template("user_rec.html", user_data=user_data, user_id=user_id, title_rec=title_rec, auth_rec=auth_rec)


if __name__ == '__main__':
    app.run(debug=True)
